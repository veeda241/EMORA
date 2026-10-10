# Decision-level fusion — results

- Trained on split `train` (n=468); evaluated on `test` (n=643).
- Encoders frozen; only the fusion head is learned. Test split never used for fitting.

## Accuracy / macro-F1

| model | accuracy | macro-F1 |
|---|---|---|
| **Learned fusion (logreg)** | **0.4774** | **0.3927** |
| standalone_text | 0.4666 | 0.3142 |
| standalone_audio | 0.4230 | 0.2943 |
| standalone_video | 0.2333 | 0.1052 |
| fixed_weight_average | 0.4946 | 0.3385 |

## Fusion classification report (test)

```
              precision    recall  f1-score   support

       angry     0.5286    0.2256    0.3162       164
     disgust     0.1944    0.3784    0.2569        37
     fearful     0.0612    0.1154    0.0800        26
       happy     0.5561    0.6770    0.6106       161
         sad     0.6250    0.3913    0.4813       115
   surprised     0.5380    0.7071    0.6111       140

    accuracy                         0.4774       643
   macro avg     0.4172    0.4158    0.3927       643
weighted avg     0.5167    0.4774    0.4707       643

```

## Interpretation

- Learned fusion macro-F1 **0.3927** vs strongest standalone (standalone_text) 0.3142 -> +0.0784.
- Learned fusion accuracy **0.4774** vs strongest standalone 0.4666 -> +0.0109.
- vs fixed-weight average: macro-F1 +0.0542, accuracy -0.0171. The learned head's main gain is balanced (macro) performance across emotions, not peak accuracy.

## Limitations

- Small fitting set: n_train=468 synchronized (text+audio+video) MELD utterances; rare classes are thin (train disgust=14, fearful=25). Weights are provisional.
- Every row requires ALL THREE modalities on the same clip; utterances where MediaPipe found no frontal face were dropped (never fabricated), which shrinks the usable set and biases it toward face-visible clips.
- Live video features come from a MediaPipe->FACET4.2 approximation (commercial FACET emotion channels are zeroed); standalone video is weak (acc 0.233, macro-F1 0.105) and mainly contributes noise the fusion must learn to down-weight.
- Test set is a single held-out MELD split; deltas of a few points are within noise. class_weight='balanced' was fixed a priori (a coach must not miss rare emotions), not tuned on test.
