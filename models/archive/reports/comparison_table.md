# Modality Model Comparison (MELD-based Stress/Burnout Tracker)

All three encoders are trained **separately first**, then projected to a common embedding dimension before the Cross-Modal Attention Transformer fuses them.

## Comparison Table

| Modality | Model | Fit | Critical limitation | Baseline accuracy | Best metric for the project |
|---|---|---|---|---|---|
| Voice | emotion2vec | Emotion-focused self-supervised speech representation; the primary voice model for this comparison. | Requires a compatible emotion2vec checkpoint and supervised classification head for the target labels. | Primary candidate | Macro F1 and weighted F1 |
| Voice | WavLM-Base | Speaker-aware self-supervised voice encoder for robust speech representations in varied acoustic conditions. | Downstream emotion classification still requires a supervised head and fine-tuning. | Comparison candidate | Macro F1 and weighted F1 |
| Voice | Wav2Vec2-Base | Self-supervised encoder that captures involuntary micro-tremors, breath shifts, pitch and harmonic variation. | Only works while the user speaks; vulnerable to background noise. | Baseline candidate | Macro F1 and weighted F1 |
| Face | MAE-DFER | Masked autoencoder pretraining adapted for dynamic facial expression recognition; strongest overall candidate. | Requires facial-expression video pretraining or a compatible checkpoint before supervised fine-tuning. | Excellent | F1-score and balanced accuracy |
| Face | EST | Expression-aware spatiotemporal transformer candidate with strong dynamic facial-expression modeling. | High compute and data requirements; implementation and checkpoint selection must be finalized. | Excellent | F1-score and temporal robustness |
| Face | Hybrid CNN + ViT | Combines local facial texture extraction from a CNN with global patch relationships from a Vision Transformer. | Performance depends on the exact CNN, ViT, fusion head, and training implementation. | Good | F1-score, accuracy, and latency |
| Text | BERT-Base-Uncased | Great general-purpose sentiment/connotation probe; good context retention on conversational transcripts. | Struggles with sarcasm and heavy informal remarks in Friends dialogue. | ~80-88% (text sentiment benchmarks) | F1-score |
| Text | RoBERTa-Base | Drops next-sentence prediction and trains on longer masked span fills; usually lifts text F1 by ~2-4% over BERT. | Slightly heavier than BERT; still an encoder-only model. | ~82-90% | F1-score |
| Text | DeBERTa-v3-Base | Decoupled attention over word positions maps conversational nuance, sarcasm, and blended emotions best of the three. | Largest of the text trio; needs the most VRAM / GPU time. | ~84-92% | F1-score |

## One-line verdict

For a **non-contact, real-time** stress tracker, this project should:

1. Use **emotion2vec** as the primary *voice* stream.
2. Keep **BERT/DeBERTa** for the *text log* stream (best chronic, cognitive-burnout signal).
3. Use **MAE-DFER** for the *face* stream, with **EST** as the alternative and **Hybrid CNN + ViT** as the third candidate.
4. Fuse all three through the **Cross-Modal Attention Transformer** so the wearable/classifier conflict is resolved by reliability weighting instead of majority vote.

## Recommended encoder choice

| Task | Primary encoder | Runner-up |
|---|---|---|
| Voice | emotion2vec | WavLM-Base / Wav2Vec2-Base |
| Face | MAE-DFER | EST / Hybrid CNN + ViT |
| Text | DeBERTa-v3-Base | RoBERTa-Base |

## Datasets for context

The datasets below are useful for **stress-specific validation** and for verifying that the Cross-Modal Transformer weights generalise beyond MELD. They are mapped to the modalities they best support: text, audio, and video/face.
### GoEmotions

- Scale: ~1.4M comments
- Strengths: Fine-grained emotion labels for conversational text
- Ideal component focus: text
- Source: Google Research
- Download: github.com/google-research/google-research/tree/master/goemotions

### MELD

- Scale: 13,000+ utterances
- Strengths: Natural multi-party conversational flow with text, audio, and video/face potential
- Ideal component focus: text | audio | video/face
- Source: Applied Research Center, CMU (CoT, 2019)
- Download: github.com/rahulbabu/meld

### RAVDESS

- Scale: 1,440 speech clips (24 actors)
- Strengths: Clean acted speech with 8 emotion codes; good for audio-only emotion comparison
- Ideal component focus: audio
- Source: Ryerson University; Zenodo mirror used here: zenodo.org/records/11063852
- Download: zenodo.org/records/11063852

### CMU-MOSEI

- Scale: 23,500+ clips
- Strengths: Large scale multimodal clips with fine-grained intensity values
- Ideal component focus: video/face | audio | text
- Source: Carnegie Mellon University
- Download: kaggle.com/cmu-mosei

