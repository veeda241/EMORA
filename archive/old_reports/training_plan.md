# Multimodal Training Plan (MELD-based Stress/Burnout Tracker)

Each modality is trained independently first. Every encoder emits a
fixed 512-dimensional vector, so the Cross-Modal Attention
Transformer can fuse them on equal terms without shape mismatches.

## Prerequisites

- Python 3.10+, PyTorch 2.x, Hugging Face `transformers`
- GoEmotions (text), MELD (audio), and CMU-MOSEI (video/features)
- Project-wide projection target: `PROJECT_DIM = 512`

## Per-modality PyTorch streams

### Voice Stream
#### emotion2vec (emotion2vec)
```python
def voice_stream(voice_model: torch.nn.Module, voice_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the voice-specific encoder and project to 512-dim vector."""
    model = voice_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> voice samples)
    # Step 1 -> Extract fixed-size audio embeddings from emotion2vec, WavLM, or Wav2Vec2 on the audio tracks, then time-pool them before projection. emotion2vec is primary, WavLM is the comparison model, and Wav2Vec2 is the baseline.
    # Step 2 -> collect hidden-state vectors into a 768-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(768, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return voice_model
```

#### WavLM-Base (wavlm-base)
```python
def voice_stream(voice_model: torch.nn.Module, voice_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the voice-specific encoder and project to 512-dim vector."""
    model = voice_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> voice samples)
    # Step 1 -> Extract fixed-size audio embeddings from emotion2vec, WavLM, or Wav2Vec2 on the audio tracks, then time-pool them before projection. emotion2vec is primary, WavLM is the comparison model, and Wav2Vec2 is the baseline.
    # Step 2 -> collect hidden-state vectors into a 768-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(768, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return voice_model
```

#### Wav2Vec2-Base (wav2vec2-base)
```python
def voice_stream(voice_model: torch.nn.Module, voice_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the voice-specific encoder and project to 512-dim vector."""
    model = voice_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> voice samples)
    # Step 1 -> Extract fixed-size audio embeddings from emotion2vec, WavLM, or Wav2Vec2 on the audio tracks, then time-pool them before projection. emotion2vec is primary, WavLM is the comparison model, and Wav2Vec2 is the baseline.
    # Step 2 -> collect hidden-state vectors into a 768-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(768, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return voice_model
```

### Face Stream
#### MAE-DFER (mae-dfer)
```python
def face_stream(face_model: torch.nn.Module, face_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the face-specific encoder and project to 512-dim vector."""
    model = face_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> face samples)
    # Step 1 -> Preprocess aligned face clips, then extract spatiotemporal features using MAE-DFER, EST, or the Hybrid CNN + ViT encoder.
    # Step 2 -> collect hidden-state vectors into a 512-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(512, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return face_model
```

#### EST (est)
```python
def face_stream(face_model: torch.nn.Module, face_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the face-specific encoder and project to 512-dim vector."""
    model = face_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> face samples)
    # Step 1 -> Preprocess aligned face clips, then extract spatiotemporal features using MAE-DFER, EST, or the Hybrid CNN + ViT encoder.
    # Step 2 -> collect hidden-state vectors into a 512-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(512, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return face_model
```

#### Hybrid CNN + ViT (hybrid-cnn-vit)
```python
def face_stream(face_model: torch.nn.Module, face_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the face-specific encoder and project to 512-dim vector."""
    model = face_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> face samples)
    # Step 1 -> Preprocess aligned face clips, then extract spatiotemporal features using MAE-DFER, EST, or the Hybrid CNN + ViT encoder.
    # Step 2 -> collect hidden-state vectors into a 512-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(512, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return face_model
```

### Text Stream
#### BERT-Base-Uncased (bert-base-uncased)
```python
def text_stream(text_model: torch.nn.Module, text_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the text-specific encoder and project to 512-dim vector."""
    model = text_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> text samples)
    # Step 1 -> Fine-tune the encoder on MELD transcripts, take the [CLS]/mean pooled hidden state.
    # Step 2 -> collect hidden-state vectors into a 768-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(768, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return text_model
```

#### RoBERTa-Base (roberta-base)
```python
def text_stream(text_model: torch.nn.Module, text_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the text-specific encoder and project to 512-dim vector."""
    model = text_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> text samples)
    # Step 1 -> Fine-tune the encoder on MELD transcripts, take the [CLS]/mean pooled hidden state.
    # Step 2 -> collect hidden-state vectors into a 768-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(768, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return text_model
```

#### DeBERTa-v3-Base (deberta-v3-base)
```python
def text_stream(text_model: torch.nn.Module, text_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:
    """Train the text-specific encoder and project to 512-dim vector."""
    model = text_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCELoss()

    # Step 0 -> dataset/parsing (E.g. MELD CSV -> text samples)
    # Step 1 -> Fine-tune the encoder on MELD transcripts, take the [CLS]/mean pooled hidden state.
    # Step 2 -> collect hidden-state vectors into a 768-dim tensor
    # Step 3 -> project to a uniform 512-dim tensor
    nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
    ),
    # Step 4 -> classification head
    nn.Linear(768, 1)
    nn.Sigmoid()
    # Step 5 -> self-supervised + supervised training loop, save best checkpoint
    return text_model
```

## Step 6 -> Cross-Modal Attention Transformer

After the three streams above return a 512-dim vector each,
stack them, normalize, and pass them through your multi-head cross-
attention transformer encoder. The attention weights decode which
modality is most reliable for the current sample (e.g. voice outranks
face when the user is speaking, face outranks text when the user
is silent).

## Suggested baselines for the comparison table

| Modality | Primary encoder | Runner-up |
|---|---|---|
| Voice | emotion2vec | WavLM-Base / Wav2Vec2-Base |
| Face | MAE-DFER | EST / Hybrid CNN + ViT |
| Text | DeBERTa-v3-Base | RoBERTa-Base |

## Metrics to record across the table

- **Voice**: inference latency + intensity tracking (Wav2Vec2)
- **Face**: classification accuracy under angle/lighting change
- **Text**: F1-score on conversational transcripts
- **Fused**: final stress score accuracy after the Cross-Modal Transformer

## Expected result

Individually the three encoders land in the 74-92% range. Once fused
through the Cross-Modal Attention Transformer the stress/burnout signal
typically rises past the 85-90% accuracy ceiling that any single
modality can reach alone.

## Notes on the datasets

- **MELD** is the main supervised source: natural multi-party
  conversational flow that keeps BERT context retention sharp.
- **CMU-MOSEI** adds 23,500+ clips with fine-grained intensity values
  to tune the Cross-Modal Transformer weighting rules.
- **GoEmotions** supplies the fine-grained text emotion labels; its
  multilabel annotations must be handled explicitly by the text model.

## Data preparation checklist

1. Split MELD into train/val/test (speaker-independent).
2. Align text transcripts, face crops and .wav tracks by utterance
   timestamp so stream 1/2/3 see the same sample index.
3. Freeze the audio/visual encoders initially; only train the
   projection head + cross-modal attention first (avoids catastrophic
   forgetting).
4. Apply modality dropout (e.g. randomly mask the audio stream in 10%
   of batches) so the Transformer learns to re-weight the remaining
   modalities during real-world sensor dropouts.

## Output artifacts

- `models/comparison_table.md` - the model comparison table
- `models/comparison_table.json` - machine-readable table snapshot
- `models/{modality}_stream.py` - each PyTorch encoder stream
- `models/cross_modal_attention_transformer.py` - fusion block
