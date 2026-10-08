# Modality and Fusion Ablation

| Model | Accuracy | Macro F1 | Status | Notes |
|---|---:|---:|---|---|
| Majority | 0.4408671388710024 | 0.0 | measured | test-set class prior |
| Text | 0.26572225799527793 | 0.2105172674808393 | measured | balanced logistic regression on temporal mean features |
| Audio | 0.22773127280532304 | 0.1782918622235927 | measured | balanced logistic regression on temporal mean features |
| Video | 0.2927666881305001 | 0.21192105282347293 | measured | balanced logistic regression on temporal mean features |
| Text + Audio | 0.2953423481433784 | 0.23019269577800364 | measured | balanced logistic regression on temporal mean features |
| Text + Video | 0.3408456750375617 | 0.2518384168679836 | measured | balanced logistic regression on temporal mean features |
| Audio + Video | 0.30113758317235456 | 0.22825704820090142 | measured | balanced logistic regression on temporal mean features |
| Simple Concatenation | 0.34728482506975744 | 0.25763670500243335 | measured | balanced logistic regression on temporal mean features |
| MFN | 0.47177505902554195 | 0.27173548383205287 | measured | saved MFN checkpoint |
| GloVe + COVAREP + HybridCNNViT + MFN | 0.4518136939257351 | 0.23786240106302314 | measured | Experiment B; frozen canonical Hybrid CNN-ViT 512-D embedding |
| GloVe + COVAREP + temporal HybridCNNViT + MFN | 0.4599699506331831 | 0.2398372623178865 | measured | Experiment C; frozen Hybrid CNN-ViT temporal (16x256) features, no broadcast |
| LSTM Fusion | 0.44043786220218933 | 0.26321067973543977 | measured | LSTM temporal fusion using GloVe + COVAREP + FACET with the same CMU-MOSEI split and evaluation protocol as MFN. |
