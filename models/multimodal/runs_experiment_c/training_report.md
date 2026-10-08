# Experiment C: GloVe + COVAREP + temporal Hybrid CNN-ViT -> MFN

```json
{
  "experiment": "C",
  "architecture": "GloVe + COVAREP + frozen HybridCNNViT temporal (B,16,256) + trainable per-step projection + MFN",
  "hybrid_temporal_dim": 256,
  "tensor_flow": {
    "text_raw": "(B, 16, 300)",
    "audio_raw": "(B, 16, 74)",
    "facet_raw": "(B, 16, 35)",
    "hybrid_temporal": "(B, 16, 256)",
    "video_projected": "(B, 16, 128)",
    "text_projected": "(B, 16, 256)",
    "audio_projected": "(B, 16, 128)",
    "mfn_input": "three aligned sequences, T=16",
    "logits": "(B, 6)"
  },
  "training": {
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "cuda",
    "weighted_sampler": true,
    "class_weighted_loss": true,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 5.0,
    "selection_metric": "validation accuracy"
  },
  "reproduce_command": "python models/multimodal/train_experiment_c.py --epochs 30 --batch-size 64 --lr 0.001 --device cuda",
  "checkpoint": "C:\\hackathon\\Gemini_CLI\\LLM-based-conversational-agents-for-behaviour-change-support\\models\\multimodal\\runs_experiment_c\\best_model_glove_covarep_hybrid_cnn_vit_temporal.pt",
  "best_val_acc": 0.4206306787814003,
  "test_loss": 8.585936553899401,
  "test_accuracy": 0.4599699506331831,
  "test_macro_f1": 0.2398372623178865,
  "test_weighted_f1": 0.4495704300035891,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.2994501055521264,
      "train_acc": 0.6393824286239431,
      "val_loss": 3.0570380657130802,
      "val_acc": 0.28861571352218063,
      "val_macro_f1": 0.19938963572298507
    },
    {
      "epoch": 2,
      "train_loss": 0.1298416766896433,
      "train_acc": 0.7811542703100111,
      "val_loss": 3.9833517117910624,
      "val_acc": 0.37359700694815606,
      "val_macro_f1": 0.23075436225501425
    },
    {
      "epoch": 3,
      "train_loss": 0.10348043159141565,
      "train_acc": 0.8184658742801127,
      "val_loss": 4.002474192818622,
      "val_acc": 0.33083912346338856,
      "val_macro_f1": 0.22189339531482585
    },
    {
      "epoch": 4,
      "train_loss": 0.0951413920135741,
      "train_acc": 0.8354368337213577,
      "val_loss": 4.281387389757492,
      "val_acc": 0.3949759486905398,
      "val_macro_f1": 0.23010795626066174
    },
    {
      "epoch": 5,
      "train_loss": 0.08720332548622992,
      "train_acc": 0.845668422987379,
      "val_loss": 4.851153341351442,
      "val_acc": 0.36611437733832175,
      "val_macro_f1": 0.23068607950491274
    },
    {
      "epoch": 6,
      "train_loss": 0.07144822239978044,
      "train_acc": 0.8676020095576522,
      "val_loss": 5.146224259947915,
      "val_acc": 0.3639764831640834,
      "val_macro_f1": 0.2227531548452839
    },
    {
      "epoch": 7,
      "train_loss": 0.06519822144008795,
      "train_acc": 0.8661316015194216,
      "val_loss": 5.493341634007203,
      "val_acc": 0.40032068412613575,
      "val_macro_f1": 0.24339579663374236
    },
    {
      "epoch": 8,
      "train_loss": 0.06174945012488716,
      "train_acc": 0.8778335988236736,
      "val_loss": 5.235236120504341,
      "val_acc": 0.37520042757883487,
      "val_macro_f1": 0.2286048741349677
    },
    {
      "epoch": 9,
      "train_loss": 0.050832610822138385,
      "train_acc": 0.8919862761916432,
      "val_loss": 5.7713361108831895,
      "val_acc": 0.4168893639764832,
      "val_macro_f1": 0.24697509203843326
    },
    {
      "epoch": 10,
      "train_loss": 0.049603509446570666,
      "train_acc": 0.8957235632888126,
      "val_loss": 5.8141038303130586,
      "val_acc": 0.3976483164083378,
      "val_macro_f1": 0.24087515386593272
    },
    {
      "epoch": 11,
      "train_loss": 0.04718658808045224,
      "train_acc": 0.8987869133684597,
      "val_loss": 6.001724802574839,
      "val_acc": 0.40032068412613575,
      "val_macro_f1": 0.2428998941175657
    },
    {
      "epoch": 12,
      "train_loss": 0.046869837920786396,
      "train_acc": 0.903872074500674,
      "val_loss": 6.303467253068754,
      "val_acc": 0.40406199893105293,
      "val_macro_f1": 0.23076488225493627
    },
    {
      "epoch": 13,
      "train_loss": 0.04391618941201546,
      "train_acc": 0.9064452885675776,
      "val_loss": 6.1733217631859425,
      "val_acc": 0.41849278460716194,
      "val_macro_f1": 0.23864541787720916
    },
    {
      "epoch": 14,
      "train_loss": 0.04032082189933048,
      "train_acc": 0.909018502634481,
      "val_loss": 6.748508125623237,
      "val_acc": 0.40406199893105293,
      "val_macro_f1": 0.2402623594690624
    },
    {
      "epoch": 15,
      "train_loss": 0.0347988287472465,
      "train_acc": 0.9201078299228036,
      "val_loss": 6.7066493888133225,
      "val_acc": 0.4083377872795297,
      "val_macro_f1": 0.24064054101120846
    },
    {
      "epoch": 16,
      "train_loss": 0.03401224273832686,
      "train_acc": 0.9208430339419189,
      "val_loss": 7.001959003145716,
      "val_acc": 0.41849278460716194,
      "val_macro_f1": 0.238195437662839
    },
    {
      "epoch": 17,
      "train_loss": 0.0353098657221148,
      "train_acc": 0.9200465629212107,
      "val_loss": 6.419519180795332,
      "val_acc": 0.39230358097274187,
      "val_macro_f1": 0.22963916509631443
    },
    {
      "epoch": 18,
      "train_loss": 0.028041279652306822,
      "train_acc": 0.9342605072907731,
      "val_loss": 7.307726978809677,
      "val_acc": 0.4051309460181721,
      "val_macro_f1": 0.2326835845200439
    },
    {
      "epoch": 19,
      "train_loss": 0.022854601101845506,
      "train_acc": 0.9431442225217498,
      "val_loss": 7.928690833117731,
      "val_acc": 0.40138963121325494,
      "val_macro_f1": 0.2322952688499208
    },
    {
      "epoch": 20,
      "train_loss": 0.022238482557275494,
      "train_acc": 0.9386104644038721,
      "val_loss": 7.759435475286732,
      "val_acc": 0.4136825227151256,
      "val_macro_f1": 0.23771081228338528
    },
    {
      "epoch": 21,
      "train_loss": 0.01880314138333455,
      "train_acc": 0.9501286607033452,
      "val_loss": 8.292370094837574,
      "val_acc": 0.3981827899518974,
      "val_macro_f1": 0.22537812924665201
    },
    {
      "epoch": 22,
      "train_loss": 0.017207568005475713,
      "train_acc": 0.955581423845117,
      "val_loss": 8.33461432097622,
      "val_acc": 0.4206306787814003,
      "val_macro_f1": 0.23308305574870322
    },
    {
      "epoch": 23,
      "train_loss": 0.014806846411201011,
      "train_acc": 0.96085038598211,
      "val_loss": 8.688297330855304,
      "val_acc": 0.40994120791020844,
      "val_macro_f1": 0.22556776464492
    },
    {
      "epoch": 24,
      "train_loss": 0.014839603860858776,
      "train_acc": 0.9596863129518441,
      "val_loss": 8.635463760983555,
      "val_acc": 0.4061998931052913,
      "val_macro_f1": 0.22419862759655917
    },
    {
      "epoch": 25,
      "train_loss": 0.013686324508089038,
      "train_acc": 0.9645876730792795,
      "val_loss": 8.802113051340582,
      "val_acc": 0.41635489043292356,
      "val_macro_f1": 0.22869023958380894
    },
    {
      "epoch": 26,
      "train_loss": 0.012299110139040208,
      "train_acc": 0.9669770861414042,
      "val_loss": 9.139051336709475,
      "val_acc": 0.40299305184393375,
      "val_macro_f1": 0.22533169241355852
    },
    {
      "epoch": 27,
      "train_loss": 0.01181650718149447,
      "train_acc": 0.9682024261732631,
      "val_loss": 9.334882220120942,
      "val_acc": 0.41047568145376806,
      "val_macro_f1": 0.2280543859890886
    },
    {
      "epoch": 28,
      "train_loss": 0.010604014312064453,
      "train_acc": 0.9733488543070702,
      "val_loss": 9.400011974499865,
      "val_acc": 0.40994120791020844,
      "val_macro_f1": 0.22922852093039206
    },
    {
      "epoch": 29,
      "train_loss": 0.010069269873977854,
      "train_acc": 0.974696728342115,
      "val_loss": 9.435896281186812,
      "val_acc": 0.40887226082308925,
      "val_macro_f1": 0.22978178731120058
    },
    {
      "epoch": 30,
      "train_loss": 0.009279102945649848,
      "train_acc": 0.9767798063962749,
      "val_loss": 9.712423651311438,
      "val_acc": 0.41047568145376806,
      "val_macro_f1": 0.2259868223069135
    }
  ]
}
```

## Per-class report

```
              precision    recall  f1-score   support

       angry       0.42      0.50      0.46      1453
     disgust       0.17      0.11      0.14       342
     fearful       0.05      0.03      0.04       179
       happy       0.62      0.62      0.62      2054
         sad       0.19      0.19      0.19       551
   surprised       0.00      0.00      0.00        80

    accuracy                           0.46      4659
   macro avg       0.24      0.24      0.24      4659
weighted avg       0.44      0.46      0.45      4659

```
