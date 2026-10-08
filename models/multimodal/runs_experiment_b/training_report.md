# Experiment B: GloVe + COVAREP + Hybrid CNN-ViT -> MFN

```json
{
  "experiment": "B",
  "architecture": "GloVe + COVAREP + frozen HybridCNNViT emb(512) + trainable projection + MFN",
  "hybrid_embedding_dim": 512,
  "checkpoint": "C:\\hackathon\\Gemini_CLI\\LLM-based-conversational-agents-for-behaviour-change-support\\models\\multimodal\\runs_experiment_b\\best_model_glove_covarep_hybrid_cnn_vit.pt",
  "best_val_acc": 0.413148049171566,
  "test_loss": 6.686476768969364,
  "test_accuracy": 0.4518136939257351,
  "test_macro_f1": 0.23786240106302314,
  "test_weighted_f1": 0.4333125765433986,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.24681477035363733,
      "train_acc": 0.6761426295797084,
      "val_loss": 3.115597877191516,
      "val_acc": 0.3853554249064671,
      "val_macro_f1": 0.22840663710378506
    },
    {
      "epoch": 2,
      "train_loss": 0.10632358278931757,
      "train_acc": 0.8239186374218845,
      "val_loss": 3.780227560974072,
      "val_acc": 0.3639764831640834,
      "val_macro_f1": 0.2309547014994222
    },
    {
      "epoch": 3,
      "train_loss": 0.08403616348018345,
      "train_acc": 0.8570640852836662,
      "val_loss": 4.167411459377907,
      "val_acc": 0.37359700694815606,
      "val_macro_f1": 0.2361697939878891
    },
    {
      "epoch": 4,
      "train_loss": 0.07694407897484301,
      "train_acc": 0.8651513294939346,
      "val_loss": 4.526750583944316,
      "val_acc": 0.3730625334045965,
      "val_macro_f1": 0.23144774268883753
    },
    {
      "epoch": 5,
      "train_loss": 0.06616187067078183,
      "train_acc": 0.8810807499080995,
      "val_loss": 4.636558660907837,
      "val_acc": 0.35809727418492787,
      "val_macro_f1": 0.21906409163710228
    },
    {
      "epoch": 6,
      "train_loss": 0.05924347110960855,
      "train_acc": 0.8872687170689867,
      "val_loss": 4.870101460029199,
      "val_acc": 0.3639764831640834,
      "val_macro_f1": 0.22102475370604382
    },
    {
      "epoch": 7,
      "train_loss": 0.049400558402368566,
      "train_acc": 0.8989707143732386,
      "val_loss": 5.0546508577021605,
      "val_acc": 0.364510956707643,
      "val_macro_f1": 0.21843957319388996
    },
    {
      "epoch": 8,
      "train_loss": 0.04522683372112367,
      "train_acc": 0.9099987746599681,
      "val_loss": 4.896593235572476,
      "val_acc": 0.3901656867985035,
      "val_macro_f1": 0.23470170928578563
    },
    {
      "epoch": 9,
      "train_loss": 0.04028763755507317,
      "train_acc": 0.9140423967651023,
      "val_loss": 5.572282288814852,
      "val_acc": 0.3853554249064671,
      "val_macro_f1": 0.223276744404278
    },
    {
      "epoch": 10,
      "train_loss": 0.03757006999421552,
      "train_acc": 0.9185148878813871,
      "val_loss": 5.564883925072462,
      "val_acc": 0.40299305184393375,
      "val_macro_f1": 0.229089904227033
    },
    {
      "epoch": 11,
      "train_loss": 0.03367023216189597,
      "train_acc": 0.9238451170199731,
      "val_loss": 6.370739426095494,
      "val_acc": 0.38482095136290756,
      "val_macro_f1": 0.22115964501868168
    },
    {
      "epoch": 12,
      "train_loss": 0.036782775648260525,
      "train_acc": 0.9155740718049259,
      "val_loss": 5.9361002008141455,
      "val_acc": 0.39283805451630144,
      "val_macro_f1": 0.22483155691317233
    },
    {
      "epoch": 13,
      "train_loss": 0.03251625073170081,
      "train_acc": 0.9269084670996202,
      "val_loss": 6.314448087117304,
      "val_acc": 0.4035275253874933,
      "val_macro_f1": 0.22876970269355068
    },
    {
      "epoch": 14,
      "train_loss": 0.032093094379045756,
      "train_acc": 0.9281950741330719,
      "val_loss": 6.341725756942366,
      "val_acc": 0.3912346338856227,
      "val_macro_f1": 0.22345328723409866
    },
    {
      "epoch": 15,
      "train_loss": 0.029685359559117865,
      "train_acc": 0.9346281093003308,
      "val_loss": 6.544101966121883,
      "val_acc": 0.38375200427578837,
      "val_macro_f1": 0.21848737558001227
    },
    {
      "epoch": 16,
      "train_loss": 0.024137522910756177,
      "train_acc": 0.9416738144835192,
      "val_loss": 7.034111714120984,
      "val_acc": 0.3901656867985035,
      "val_macro_f1": 0.2250185702531049
    },
    {
      "epoch": 17,
      "train_loss": 0.0270240726281492,
      "train_acc": 0.934444308295552,
      "val_loss": 6.7000818489580345,
      "val_acc": 0.413148049171566,
      "val_macro_f1": 0.21870856156945737
    },
    {
      "epoch": 18,
      "train_loss": 0.025788827056840123,
      "train_acc": 0.937568925376792,
      "val_loss": 7.050539807799281,
      "val_acc": 0.4061998931052913,
      "val_macro_f1": 0.22349530414019933
    },
    {
      "epoch": 19,
      "train_loss": 0.02135041324521503,
      "train_acc": 0.9488420536698934,
      "val_loss": 7.336127779943812,
      "val_acc": 0.40780331373597006,
      "val_macro_f1": 0.22422392147665504
    },
    {
      "epoch": 20,
      "train_loss": 0.020583660667066038,
      "train_acc": 0.9494547236858228,
      "val_loss": 7.535766133135618,
      "val_acc": 0.38588989845002675,
      "val_macro_f1": 0.21700106083045467
    },
    {
      "epoch": 21,
      "train_loss": 0.020231082140778577,
      "train_acc": 0.9538046807989217,
      "val_loss": 7.490342144275464,
      "val_acc": 0.3997862105825762,
      "val_macro_f1": 0.22457772902731554
    },
    {
      "epoch": 22,
      "train_loss": 0.01975083916557506,
      "train_acc": 0.9522730057590981,
      "val_loss": 7.647933950276378,
      "val_acc": 0.39070016034206306,
      "val_macro_f1": 0.22183083445621357
    },
    {
      "epoch": 23,
      "train_loss": 0.014915552166295117,
      "train_acc": 0.9629334640362701,
      "val_loss": 7.812559061672266,
      "val_acc": 0.3981827899518974,
      "val_macro_f1": 0.2228268716969787
    },
    {
      "epoch": 24,
      "train_loss": 0.014826325214226126,
      "train_acc": 0.9648940080872442,
      "val_loss": 8.056752615724255,
      "val_acc": 0.4051309460181721,
      "val_macro_f1": 0.2250929720136822
    },
    {
      "epoch": 25,
      "train_loss": 0.012808715546088955,
      "train_acc": 0.9663644161254749,
      "val_loss": 8.411581124041184,
      "val_acc": 0.396044895777659,
      "val_macro_f1": 0.21474951380051707
    },
    {
      "epoch": 26,
      "train_loss": 0.011619230427039997,
      "train_acc": 0.9709594412449455,
      "val_loss": 8.48032040904385,
      "val_acc": 0.39551042223409943,
      "val_macro_f1": 0.21704332812062513
    },
    {
      "epoch": 27,
      "train_loss": 0.010925276414938082,
      "train_acc": 0.9726136502879549,
      "val_loss": 8.565503035300177,
      "val_acc": 0.40299305184393375,
      "val_macro_f1": 0.2202657468009098
    },
    {
      "epoch": 28,
      "train_loss": 0.00991516212573634,
      "train_acc": 0.974696728342115,
      "val_loss": 8.80567526141951,
      "val_acc": 0.4083377872795297,
      "val_macro_f1": 0.2190371826075187
    },
    {
      "epoch": 29,
      "train_loss": 0.009983000879575035,
      "train_acc": 0.9748805293468937,
      "val_loss": 8.920284656074072,
      "val_acc": 0.4035275253874933,
      "val_macro_f1": 0.21735240093039643
    },
    {
      "epoch": 30,
      "train_loss": 0.00944438127875941,
      "train_acc": 0.9758608013723808,
      "val_loss": 9.045922182512308,
      "val_acc": 0.40299305184393375,
      "val_macro_f1": 0.21975709475766306
    }
  ]
}
```

## Per-class report

```
              precision    recall  f1-score   support

       angry       0.43      0.33      0.38      1453
     disgust       0.16      0.10      0.12       342
     fearful       0.09      0.03      0.05       179
       happy       0.57      0.70      0.63      2054
         sad       0.21      0.27      0.24       551
   surprised       0.02      0.01      0.02        80

    accuracy                           0.45      4659
   macro avg       0.25      0.24      0.24      4659
weighted avg       0.43      0.45      0.43      4659

```
