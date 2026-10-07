# Modality embedding shapes (what the Cross-Modal Transformer consumes)

Each encoder is trained independently, then a Linear + LayerNorm +
GELU projection maps it to a uniform `PROJECT_DIM` vector before the
cross-attention fusion.

| Modality | Encoder | Pre-projection dim | Projected dim | Transformer input dim |
|---|---|---|---|---|
| Voice | emotion2vec | 768 | 512 | 512 |
| Voice | WavLM-Base | 768 | 512 | 512 |
| Voice | Wav2Vec2-Base | 768 | 512 | 512 |
| Face | MAE-DFER | 512 | 512 | 512 |
| Face | EST | 512 | 512 | 512 |
| Face | Hybrid CNN + ViT | 512 | 512 | 512 |
| Text | BERT-Base-Uncased | 768 | 512 | 512 |
| Text | RoBERTa-Base | 768 | 512 | 512 |
| Text | DeBERTa-v3-Base | 768 | 512 | 512 |

## End-to-end flow

```text
MELD utterances                        MELD frames                    MELD audio
  |  text transcripts                  |  video frames (MediaPipe     |  .wav ->
  v                                    v                            v
 BERT/RoBERTa/DeBERTa                 MAE-DFER / EST / CNN+ViT       emotion2vec / WavLM / Wav2Vec2
  |  [CLS]/mean pool                 |  GAP pool                   |  time-pooled block
  v                                    v                            v
 768-dim vector                     512-dim vector                 768-dim vector
  |  Linear(768 -> 512)              |  Linear(512 -> 512)          |  Linear(768 -> 512)
  v                                    v                            v
 512-dim projected                  512-dim projected                512-dim projected
  |  concat -> layer-norm            |                              |
  v                                    v                            v
  [v_face | v_text | v_voice] ------> Multi-head Cross-Attention ----> Stress score
```

## Sample tensor shapes

### Voice - emotion2vec
```python
model = voice_model  # e.g. Wav2Vec2Model
x = sample_voice(model, batch)           # (2, 768)
proj = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Voice - WavLM-Base
```python
model = voice_model  # e.g. Wav2Vec2Model
x = sample_voice(model, batch)           # (2, 768)
proj = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Voice - Wav2Vec2-Base
```python
model = voice_model  # e.g. Wav2Vec2Model
x = sample_voice(model, batch)           # (2, 768)
proj = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Face - MAE-DFER
```python
model = face_model  # e.g. Wav2Vec2Model
x = sample_face(model, batch)           # (2, 512)
proj = nn.Sequential(
    nn.Linear(512, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Face - EST
```python
model = face_model  # e.g. Wav2Vec2Model
x = sample_face(model, batch)           # (2, 512)
proj = nn.Sequential(
    nn.Linear(512, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Face - Hybrid CNN + ViT
```python
model = face_model  # e.g. Wav2Vec2Model
x = sample_face(model, batch)           # (2, 512)
proj = nn.Sequential(
    nn.Linear(512, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Text - BERT-Base-Uncased
```python
model = text_model  # e.g. Wav2Vec2Model
x = sample_text(model, batch)           # (2, 768)
proj = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Text - RoBERTa-Base
```python
model = text_model  # e.g. Wav2Vec2Model
x = sample_text(model, batch)           # (2, 768)
proj = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
### Text - DeBERTa-v3-Base
```python
model = text_model  # e.g. Wav2Vec2Model
x = sample_text(model, batch)           # (2, 768)
proj = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
)
v = proj(x)                             # (2, 512)
print(v.shape)                          # torch.Size([2, 512])
```
