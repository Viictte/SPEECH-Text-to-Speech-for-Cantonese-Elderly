# SPEECH: Specialized Preference-Enhanced Elderly Cantonese TTS

**SPEECH** is a text-to-speech (TTS) framework designed to improve communication accessibility for elderly Cantonese speakers with hearing impairments. By combining a latent diffusion backbone with direct human feedback, SPEECH delivers culturally relevant, personalized, and highly intelligible Cantonese speech.

---

## 🌟 Key Features

- **Elderly-Centric Acoustic Supervising (ECAS)**  
  – Curated, high-quality audio–text pairs tailored for the elderly demographic  
  – Noise-reduction, precise alignment, and multi-strategy augmentation  
  – Frequency-weighted loss to emphasize 1–4 kHz band for enhanced consonant clarity  

- **Direct Preference Optimization (DPO)**  
  – Pairwise human feedback on clarity, pace, and pleasantness  
  – Fine-tuning objective that aligns synthesis with listener preferences  

- **Data-Efficient Latent Diffusion**  
  – Operates effectively on relatively small Cantonese datasets  
  – VQ-VAE encoder/decoder with HiFi-GAN vocoder  

---

## 📂 Repository Structure

