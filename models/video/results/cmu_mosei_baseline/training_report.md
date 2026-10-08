# CMU-MOSEI FACET Baseline — Training Report

## Architecture
- Input: 35-dim FACET 4.2 facial features (temporal mean-pool per utterance)
- MLP: Linear(35→256) → BN → GELU → Dropout(0.3) → Linear(256→128) → BN → GELU → Dropout(0.2) → Linear(128→6)

## Dataset
- Source: CMU-MOSEI VisualFacet42.csd
- Total samples: 3293
- Train: 2305  Val: 494  Test: 494

## Test Results

| Metric | Value |
|---|---|
| **Macro F1** | **0.0679** |
| Accuracy | 0.0628 |
| UAR | 0.2205 |
| Weighted F1 | 0.0325 |

### Per-class F1

| Class | F1 |
|---|---|
| angry | 0.1761 |
| disgust | 0.0889 |
| fearful | 0.0993 |
| happy | 0.0000 |
| sad | 0.0430 |
| surprised | 0.0000 |