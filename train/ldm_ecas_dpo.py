# src/ldm_ecas_dpo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from audioldm_train.utilities.model_util import instantiate_from_config


class LatentDiffusionWithECASDPO(pl.LightningModule):
    """
    Wraps the latent diffusion TTS model and adds:
      - ECAS loss: ASR‐based intelligibility + 1–4 kHz mel‐band MSE
      - DPO loss: preference fine‐tuning from paired human judgments
    """

    def __init__(
        self,
        cond_stage_config: dict,
        unet_config: dict,
        noise_schedule: dict,
        lambda_ecas: float = 0.1,
        lambda_dpo: float = 1.0,
        asr_model_name: str = "facebook/wav2vec2-base-960h",
        pref_hidden: int = 64,
        **unused  # catch any extra diffusion args
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) instantiate your latent-diffusion TTS backbone
        cfg = {
            "target": "audioldm_train.modules.diffusion.LatentDiffusion",
            "params": {
                "cond_stage_config": cond_stage_config,
                "unet_config": unet_config,
                "noise_schedule": noise_schedule,
            },
        }
        self.diffusion = instantiate_from_config(cfg)

        # 2) ECAS: frozen ASR for CTC loss
        self.asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name).eval()
        for p in self.asr_model.parameters():
            p.requires_grad = False

        # mel‐spectrogram (to compute freq‐band MSE)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000, n_fft=1024, win_length=1024, hop_length=256,
            n_mels=80, f_min=0.0, f_max=8000.0
        )

        # identify mel‐bin indices covering 1–4 kHz
        mel_bins = self.mel_transform.n_mels
        freq_axis = torch.linspace(0, 8000, mel_bins)
        band_mask = (freq_axis >= 1000) & (freq_axis <= 4000)
        self.register_buffer("band_mask", band_mask.float())

        # 3) DPO: small MLP scoring from average‐pooled mel
        self.pref_net = nn.Sequential(
            nn.Linear(mel_bins, pref_hidden),
            nn.ReLU(),
            nn.Linear(pref_hidden, 1),
        )

    def forward(self, *args, **kwargs):
        return self.diffusion(*args, **kwargs)

    def generate_sample(self, batch):
        """
        Return generated waveforms given input text/audio in batch.
        Assumes self.diffusion.sample(batch) -> dict with 'waveform' key.
        Adapt to your API.
        """
        out = self.diffusion.sample(batch)
        return out["waveform"]  # (B, T)

    def compute_ecas_loss(self, waveform_gen, batch):
        """
        ECAS = CTC-ASR loss + freq‐band MSE
        Expects batch["waveform"] (ground truth) and batch["text"] (list of strings).
        """
        device = waveform_gen.device
        # --- CTC ASR loss ---
        # 1) process generated
        inputs = self.asr_processor(
            waveform_gen, sampling_rate=16_000, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            logits = self.asr_model(inputs.input_values).logits  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)

        # 2) prepare labels
        texts = batch["text"]
        with self.asr_processor.as_target_processor():
            labels = self.asr_processor(texts, return_tensors="pt", padding=True).input_ids
        labels = labels.to(device)
        input_lengths = torch.full((waveform_gen.size(0),), log_probs.size(0), dtype=torch.long)
        label_lengths = (labels != self.asr_processor.tokenizer.pad_token_id).sum(dim=1)

        ctc_loss = F.ctc_loss(
            log_probs, labels, input_lengths, label_lengths,
            blank=self.asr_processor.tokenizer.pad_token_id, zero_infinity=True
        )

        # --- frequency‐weighted MSE on mel band 1-4 kHz ---
        mel_gen = self.mel_transform(waveform_gen)         # (B, M, T')
        mel_gt  = self.mel_transform(batch["waveform"].to(device))
        band = self.band_mask[None, :, None]               # (1, M, 1)
        freq_loss = F.mse_loss(mel_gen * band, mel_gt * band)

        return ctc_loss + freq_loss

    def compute_dpo_loss(self, batch):
        """
        DPO: expects batch to contain:
          - 'pref_audio_i', 'pref_audio_j' waveforms of shape (B, T)
        Computes cross‐entropy loss forcing model to score _i_ > _j_.
        """
        if "pref_audio_i" not in batch:
            return torch.tensor(0.0, device=self.device)

        wav_i = batch["pref_audio_i"].to(self.device)  # (B, T)
        wav_j = batch["pref_audio_j"].to(self.device)

        # mel + avg‐pool to get (B, M)
        mel_i = self.mel_transform(wav_i).mean(dim=-1)  # (B, M)
        mel_j = self.mel_transform(wav_j).mean(dim=-1)

        score_i = self.pref_net(mel_i).squeeze(-1)  # (B,)
        score_j = self.pref_net(mel_j).squeeze(-1)

        logits = torch.stack([score_i, score_j], dim=1)  # (B, 2)
        # label 0 means index i is preferred
        labels = torch.zeros(wav_i.size(0), dtype=torch.long, device=self.device)
        dpo_loss = F.cross_entropy(logits, labels)
        return dpo_loss

    def training_step(self, batch, batch_idx):
        # 1) base diffusion loss
        out = self.diffusion.training_step(batch, batch_idx)
        diff_loss = out["loss"]

        # 2) synthesize a batch
        generated = self.generate_sample(batch)  # (B, T)

        # 3) ECAS
        ecas = self.compute_ecas_loss(generated, batch)

        # 4) DPO
        dpo  = self.compute_dpo_loss(batch)

        loss = diff_loss + self.hparams.lambda_ecas * ecas + self.hparams.lambda_dpo * dpo

        # logging
        self.log("train/diff_loss", diff_loss, prog_bar=True)
        self.log("train/ecas_loss",  ecas,     prog_bar=True)
        self.log("train/dpo_loss",   dpo,      prog_bar=True)
        self.log("train/total_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.diffusion.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.diffusion.configure_optimizers()

