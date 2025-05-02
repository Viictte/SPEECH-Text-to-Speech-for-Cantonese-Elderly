# SPEECH: Specialized Preference-Enhanced Elderly Cantonese TTS

**SPEECH** is a text-to-speech (TTS) framework designed to improve communication accessibility for elderly Cantonese speakers with hearing impairments. By combining a latent diffusion backbone with direct human feedback, SPEECH delivers culturally relevant, personalized, and highly intelligible Cantonese speech.

---

## 🌟 Key Features

- **Elderly-Centric Acoustic Supervising (ECAS)**  
  - Curated, high-quality audio–text pairs tailored for the elderly demographic  
  - Noise reduction, precise alignment, and multi-strategy augmentation  
  - Frequency-weighted loss to emphasize the 1–4 kHz band for enhanced consonant clarity  

- **Direct Preference Optimization (DPO)**  
  - Pairwise human feedback on clarity, pace, and pleasantness  
  - Fine-tuning objective that aligns synthesis with listener preferences  

- **Data-Efficient Latent Diffusion**  
  - Operates effectively on relatively small Cantonese datasets  
  - VQ-VAE encoder/decoder with HiFi-GAN vocoder  

---

## 📂 Repository Structure
├── README.md
├── LICENSE
├── config/
│   └── 2023_08_23_reproduce_audioldm/
│       └── train_ecas_dpo.yaml      # Experimental configuration
├── data/
│   └── metadata/                     # Dataset metadata
├── loss/                             # Subjective metric implementations
├── metrics/                          # Objective & subjective metric code
│   └── subjective_metrics.py
├── preprocessing/                    # Data preprocessing pipelines
│   ├── transcribe_whisper.py
│   ├── segment_rules.py
│   └── convert_to_audiocaps.py
└── src/
├── autoencoder.py                # VAE definition
├── latent_diffusion.py           # Latent diffusion model
├── ldm_ecas_dpo.py               # ECAS & DPO loss implementations
└── train_ecas_dpo.py             # Training script

---

## 📈 Dataset

Our self-collected and augmented **Elderly Cantonese Speech Dataset** (≈ 12 020 clips, 26 GB) is publicly available:

> **Hugging Face**: [viictte/Elderly-Cantonese](https://huggingface.co/datasets/viictte/Elderly-Cantonese/tree/main)

<img src="https://github.com/user-attachments/assets/6db9fc5a-d665-4f9f-8688-1785601b46b8" alt="Dataset Overview" width="800"/>

---

## 🚀 Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/SPEECH.git
   cd SPEECH

2. **Install dependencies**
   ```python
   pip install -r requirements.txt

4. **Train model**
 ```python
python src/train_ecas_dpo.py \
  --config_yaml config/2023_08_23_reproduce_audioldm/train_ecas_dpo.yaml

```
 🧮 Metrics
	•	Objective
	•	Frechet Audio Distance (FAD)
	•	Inception Score (IS)
	•	Kullback–Leibler Divergence (KL)
	•	Subjective
	•	Overall Quality (OVL): 1–100 scale
	•	Relevance to Text (REL): 1–100 scale

🤝 Acknowledgments
	•	Whisper for robust transcription
	•	AudioLDM and NaturalSpeech 2 for latent diffusion inspiration
	•	DPO framework for human-in-the-loop optimization
