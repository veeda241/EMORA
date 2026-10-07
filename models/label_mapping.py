# Emotion Label Mapping Across Datasets

Consistent emotion label space for the multimodal stress/burnout tracker.

## Current dataset label spaces

### MELD (text/audio/video)
7 emotions from `data/raw/meld/{train,dev,test}_sent_emo.csv`:
- anger, disgust, fear, joy, neutral, sadness, surprise

### IEMOCAP (audio-first baseline)
Standard IEMOCAP 4-class scheme (most common for emotion-recognition benchmarks):
- angry, happy, neutral, sad

Some IEMOCAP papers also use a 6-class scheme:
- angry, happy, neutral, sad, excited, frustrated

### GoEmotions (text)
~1.4M comments with 27 fine-grained labels + neutral.
For fusion, these are usually collapsed to a smaller shared space.

## Recommended shared label space

For a first multimodal fusion pass, collapse to the **4-class IEMOCAP core**:
- `neutral`
- `happy` (maps to MELD `joy`)
- `sad` (maps to MELD `sadness`)
- `angry` (maps to MELD `anger`)

MELD-only emotions that do not map cleanly should be either:
- dropped from the fusion training set, or
- grouped under a broader parent class depending on the target use case.

### Mapping table (MELD -> shared 4-class)

| MELD | Shared space | Notes |
|---|---|---|
| joy | happy | direct equivalent |
| sadness | sad | direct equivalent |
| anger | angry | direct equivalent |
| neutral | neutral | direct equivalent |
| disgust | (drop / review) | not in IEMOCAP 4-class |
| fear | (drop / review) | not in IEMOCAP 4-class |
| surprise | (drop / review) | not in IEMOCAP 4-class |

## Label mapping file

A canonical Python dict lives in `models/label_mapping.py` so both the text and audio pipelines import the same mapping.

```python
# models/label_mapping.py
SHARED_EMOTION_CLASSES = ["neutral", "happy", "sad", "angry"]

MELD_TO_SHARED = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "neutral": "neutral",
    # disgust, fear, surprise intentionally omitted for the 4-class fusion space
}

IEMOCAP_TO_SHARED = {
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    # excited, frustrated omitted unless we expand the shared space
}
```

## Future expansion

If later we add the full 7-class MELD space or the 6-class IEMOCAP space,
extend `SHARED_EMOTION_CLASSES` and the two mapping dicts above. The training
scripts should always import from this file rather than hard-coding label names.
