Text-to-speech (TTS) systems play a crucial role in improving communication accessibility for
individuals with hearing impairments, particularly among elderly Cantonese speakers. However, two
major challenges persist: (1) limited availability of high-quality Cantonese resources in comparison to
Mandarin or English, and (2) insufficient personalization for elderly users, whose auditory preferences
and cultural nuances often differ from the broader population. To address these challenges, we propose
SPEECH is a specialized TTS framework that integrates a latent diffusion architecture to operate
effectively with relatively small datasets and employs Direct Preference Optimization (DPO) to
incorporate direct user feedback from elderly speakers. Our framework features two core components:
Elderly-Centric Acoustic Supervisor (ECAS) and Direct Preference Optimization. ECAS focuses
on collecting and refining a curated, high-quality audio-text pair dataset tailored for the elderly
Cantonese demographic, leveraging noise reduction, precise alignment, and diverse augmentation
strategies to capture the linguistic richness of Cantonese. And, DPO aligns the synthesized speech
with the tonal balance, pacing, and clarity preferences articulated by elderly users. By uniting a robust
data processing pipeline with feedback-driven optimization, SPEECH delivers culturally relevant,
personalized, and intelligible speech that addresses the unique communication needs of the elderly
Cantonese-speaking community.

1) Dataset can be found : [viictte/Elderly-Cantonese](https://huggingface.co/datasets/viictte/Elderly-Cantonese/tree/main)

<img width="844" alt="Screenshot 2025-05-02 at 13 46 39" src="https://github.com/user-attachments/assets/6db9fc5a-d665-4f9f-8688-1785601b46b8" />
   
2) Data Preprocessing is in the folder which consists the a) transcribing with Whisper b) segmenting the dataset based on the three rules c) encoding the file names and converting to the AudioCap form for training purposes.
   
3) The train folder consists of autoencoder.py latent_diffusion.py ldm_ecas_dpo.py train_ecas_dpo.py. The ldm_ecas_dpo.py compiles the core calculation of ECAS and DPO losses and the train_ecas_dpo.py is the training script.
4) The experimental configuration can be found in : config/2023_08_23_reproduce_audioldm/train_ecas_dpo.yaml
5) The calculation of the subjective metrics is in the loss folder.
