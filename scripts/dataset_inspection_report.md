# Dataset Inspection Report — Audio Modality

## Dataset used
- **Source folder:** `data/raw/meld/`
- **Files inspected:**
  - `train_sent_emo.csv`
  - `dev_sent_emo.csv`
  - `test_sent_emo.csv`
- **Metadata file also present:** `datasets.yaml` (contains utterance text, speaker, timestamp, emotion, sentiment — no audio files)

## What I checked
1. Directory contents under `data/raw/`
2. CSV structure for every MELD split
3. Emotion column values and distribution
4. Number of emotion classes
5. Exact emotion class names
6. Train / validation / test sizes
7. Missing emotion labels
8. Duplicate utterance IDs per split
9. Label contiguity / range validity
10. Train / validation / test overlap by utterance ID
11. Audio file presence on disk
12. Class imbalance

## Dataset format and structure
- Each row is one utterance.
- Columns include at least:
  - `Sr No.`
  - `Utterance`
  - `Speaker`
  - `Emotion`
  - `Sentiment`
  - `Dialogue_ID`
  - `Utterance_ID`
  - `Season`
  - `Episode`
  - `StartTime`
  - `EndTime`
- Emotion labels are provided as **string names**, not integer IDs.
- Split membership is defined by CSV file, not by a separate split column.
- Utterance identity key appears to be `(Dialogue_ID, Utterance_ID)`.

## Emotion classes
- **Number of emotion classes:** 7
- **Exact class names:**
  - anger
  - disgust
  - fear
  - joy
  - neutral
  - sadness
  - surprise

No other emotion labels were found in the MELD CSVs.

## Class counts
### Overall
- anger: 1607
- disgust: 361
- fear: 358
- joy: 2308
- neutral: 6436
- sadness: 1002
- surprise: 1636

### Train
- neutral: 4710
- joy: 1743
- surprise: 1205
- anger: 1109
- sadness: 683
- disgust: 271
- fear: 268
- total: 9989

### Validation
- neutral: 470
- joy: 163
- surprise: 150
- anger: 153
- sadness: 111
- fear: 40
- disgust: 22
- total: 1109

### Test
- neutral: 1256
- joy: 402
- anger: 345
- surprise: 281
- sadness: 208
- disgust: 68
- fear: 50
- total: 2610

## Label mapping
Mapping was derived automatically from the dataset by sorting unique emotion names alphabetically. It is **consistent across all splits** because it is computed from the union of all labels, not from a single split.

```
0 -> anger
1 -> disgust
2 -> fear
3 -> joy
4 -> neutral
5 -> sadness
6 -> surprise
```

`num_labels` should therefore be **7**, but the training code should still compute it from the dataset rather than hard-coding it.

## Label validation
- No missing emotion labels were found.
- Every emotion value maps cleanly to one of the 7 classes above.
- All resulting integer labels satisfy `0 <= label < num_labels` across train, dev, and test.
- No out-of-range labels were detected.

## Train / validation / test overlap
Using `(Dialogue_ID, Utterance_ID)` as utterance identity:
- train ∩ dev: overlap present
- train ∩ test: overlap present
- dev ∩ test: overlap present

This means **the split files are not strictly disjoint at the utterance-ID level as provided**. This is important for audio training because audio is usually aligned per utterance. If the same utterance ID appears in more than one split, the same audio segment could leak between splits unless deduplication or explicit split discipline is applied.

## Missing or corrupted audio files
- I searched `data/raw/` for audio files with extensions such as `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`.
- **No audio files were found** in the provided data snapshot.
- The MELD CSVs include utterance timestamps, so they are designed to align with an external MELD audio directory, but that audio directory is not present in `data/raw/` here.

This is the key blocker for audio training:
- The text labels and split structure are fully usable.
- The aligned audio segments are not present in the provided `data/raw/` tree.

## Sample rate / audio format
I could not determine audio sample rate or container format from `data/raw/` because no audio files were present. MELD audio is typically distributed separately and usually provided as WAV files aligned to the episode/utterance timestamps. If you have the MELD audio folder, I can inspect its exact format and sample rate before training.

## Class distribution
The dataset is imbalanced:
- `neutral` dominates.
- `disgust` and `fear` are the smallest classes.
- Class ratios relative to the largest class:
  - neutral: 1.000
  - joy: 0.359
  - surprise: 0.254
  - anger: 0.250
  - sadness: 0.156
  - disgust: 0.056
  - fear: 0.056

This matters for evaluation: accuracy alone will be misleading, so per-class metrics and macro F1 are important.

## Issues found
1. **No audio files present in `data/raw/`.**
   - The text-side CSVs are complete, but the corresponding audio segments are not in the provided snapshot.
   - Audio training cannot proceed until aligned audio is available.

2. **Split overlap at utterance-ID level.**
   - Train, dev, and test CSVs share some `(Dialogue_ID, Utterance_ID)` keys.
   - If audio is aligned purely by utterance ID, the same utterance could appear in multiple splits.
   - This should be resolved before training by enforcing disjoint splits.

3. **Class imbalance.**
   - Not a blocker, but it should be reflected in metrics and should be considered when interpreting results.

## Issues intentionally not found
- I did not assume 6 or 7 classes. The dataset contains exactly 7 emotion classes.
- I did not assume `neutral` is present. It is present and dominant.
- I did not assume any particular sample rate; none could be verified from the files present.

## Ready-to-train verdict
**Not ready yet for audio training**, because:
- No aligned audio files were found under `data/raw/`, and
- the provided splits overlap at the utterance level.

What is ready now:
- Dataset inspection report
- Verified emotion label set
- Verified label mapping
- Verified that labels are contiguous and in range
- A saved label mapping file: `models/audio/label_mapping.json`

## Next step before training
Place the matching MELD audio directory so each utterance can be located and loaded reliably, then confirm:
- audio file naming/alignment convention
- sample rate and format
- which episodes/folders contain the audio
- how to map `(Dialogue_ID, Utterance_ID, StartTime, EndTime)` to an actual audio segment

Once that is known, the Wav2Vec2 audio training pipeline can proceed using the label mapping already computed here.
