"""Model comparison grid for the MELD-based stress/burnout tracker.

Produces a Markdown comparison table (and a JSON snapshot) for the three
modalities the project uses on MELD:

    Text Log  -> BERT / RoBERTa / DeBERTa
    Face Stream -> MAE-DFER / EST / Hybrid CNN + ViT
    Voice Stream -> emotion2vec / WavLM / Wav2Vec2

The table mirrors what the user asked for in chat and is the single data
source the Cross-Modal Attention Transformer project consumes.
"""

from __future__ import annotations

MODEL_REGISTRY = {
    # Text models (GoEmotions / MELD transcripts -> 768-dim vector)
    "text": {
        "bert-base-uncased": {
            "name": "BERT-Base-Uncased",
            "fit": "Great general-purpose sentiment/connotation probe; good "
                   "context retention on conversational transcripts.",
            "limit": "Struggles with sarcasm and heavy informal remarks in "
                     "Friends dialogue.",
            "baseline_acc": "~80-88% (text sentiment benchmarks)",
            "target_metric": "F1-score",
            "hugging_face": "bert-base-uncased",
        },
        "roberta-base": {
            "name": "RoBERTa-Base",
            "fit": "Drops next-sentence prediction and trains on longer masked "
                   "span fills; usually lifts text F1 by ~2-4% over BERT.",
            "limit": "Slightly heavier than BERT; still an encoder-only model.",
            "baseline_acc": "~82-90%",
            "target_metric": "F1-score",
            "hugging_face": "roberta-base",
        },
        "deberta-v3-base": {
            "name": "DeBERTa-v3-Base",
            "fit": "Decoupled attention over word positions maps conversational "
                   "nuance, sarcasm, and blended emotions best of the three.",
            "limit": "Largest of the text trio; needs the most VRAM / GPU time.",
            "baseline_acc": "~84-92%",
            "target_metric": "F1-score",
            "hugging_face": "microsoft/deberta-v3-base",
        },
    },
    # Face/video models (MELD or CMU-MOSEI face/video streams -> 512-dim vector)
    "face": {
        "mae-dfer": {
            "name": "MAE-DFER",
            "fit": "Masked autoencoder pretraining adapted for dynamic facial "
                     "expression recognition; strongest overall candidate.",
            "limit": "Requires facial-expression video pretraining or a "
                       "compatible checkpoint before supervised fine-tuning.",
            "baseline_acc": "Excellent",
            "target_metric": "F1-score and balanced accuracy",
            "hugging_face": None,
            "implementation_status": "planned",
        },
        "est": {
            "name": "EST",
            "fit": "Expression-aware spatiotemporal transformer candidate with "
                     "strong dynamic facial-expression modeling.",
            "limit": "High compute and data requirements; implementation and "
                       "checkpoint selection must be finalized.",
            "baseline_acc": "Excellent",
            "target_metric": "F1-score and temporal robustness",
            "hugging_face": None,
            "implementation_status": "planned",
        },
        "hybrid-cnn-vit": {
            "name": "Hybrid CNN + ViT",
            "fit": "Combines local facial texture extraction from a CNN with "
                     "global patch relationships from a Vision Transformer.",
            "limit": "Performance depends on the exact CNN, ViT, fusion head, "
                     "and training implementation.",
            "baseline_acc": "Good",
            "target_metric": "F1-score, accuracy, and latency",
            "hugging_face": None,
            "implementation_status": "planned",
        },
    },
    # Voice/audio models (MELD .wav or RAVDESS speech -> 768-dim vector)
    "voice": {
        "emotion2vec": {
            "name": "emotion2vec",
            "fit": "Emotion-focused self-supervised speech representation; "
                     "the primary voice model for this comparison.",
            "limit": "Requires a compatible emotion2vec checkpoint and "
                       "supervised classification head for the target labels.",
            "baseline_acc": "Primary candidate",
            "target_metric": "Macro F1 and weighted F1",
            "hugging_face": None,
            "implementation_status": "planned",
            "role": "primary",
            "supported_datasets": ["meld", "ravdess"],
            "note": "Use emotion2vec as the primary audio encoder; checkpoint integration is still required.",
        },
        "wavlm-base": {
            "name": "WavLM-Base",
            "fit": "Speaker-aware self-supervised voice encoder for robust "
                     "speech representations in varied acoustic conditions.",
            "limit": "Downstream emotion classification still requires a "
                     "supervised head and fine-tuning.",
            "baseline_acc": "Comparison candidate",
            "target_metric": "Macro F1 and weighted F1",
            "hugging_face": "microsoft/wavlm-base",
            "hf_sequence_classification": "microsoft/wavlm-base",
            "supported_datasets": ["meld", "ravdess"],
            "note": "Comparison model for the primary emotion2vec audio branch.",
            "role": "comparison",
        },
        "wav2vec2-base": {
            "name": "Wav2Vec2-Base",
            "fit": "Self-supervised encoder that captures involuntary "
                     "micro-tremors, breath shifts, pitch and harmonic variation.",
            "limit": "Only works while the user speaks; vulnerable to "
                     "background noise.",
            "baseline_acc": "Baseline candidate",
            "target_metric": "Macro F1 and weighted F1",
            "hugging_face": "facebook/wav2vec2-base",
            "hf_sequence_classification": "facebook/wav2vec2-base",
            "supported_datasets": ["meld", "ravdess"],
            "note": "Wav2Vec2 is the baseline audio encoder and current implemented trainer.",
            "role": "baseline",
        },
    },
}

# Dataset metadata that the user asked for at the start of the conversation.
DATASETS = {
    "goemotions": {
        "name": "GoEmotions",
        "scale": "~1.4M comments",
        "strength": "Fine-grained emotion labels for conversational text",
        "primary_modality": "text",
        "source": "Google Research",
        "download": "github.com/google-research/google-research/tree/master/goemotions",
        "note": "Use for text encoder comparison; labels must be handled as multilabel.",
    },
    "meld": {
        "name": "MELD",
        "scale": "13,000+ utterances",
        "strength": "Natural multi-party conversational flow with text, audio, and video/face potential",
        "primary_modality": "text | audio | video/face",
        "source": "Applied Research Center, CMU (CoT, 2019)",
        "download": "github.com/rahulbabu/meld",
        "note": "Can support all three modalities if text transcripts, aligned audio, and face/video frames are prepared.",
    },
    "ravdess": {
        "name": "RAVDESS",
        "scale": "1,440 speech clips (24 actors)",
        "strength": "Clean acted speech with 8 emotion codes; good for audio-only emotion comparison",
        "primary_modality": "audio",
        "source": "Ryerson University; Zenodo mirror used here: zenodo.org/records/11063852",
        "download": "zenodo.org/records/11063852",
        "note": "Speech-only dataset; not a video dataset. Use for audio encoder comparison after excluding neutral/calm if requested.",
    },
    "cmu-mosei": {
        "name": "CMU-MOSEI",
        "scale": "23,500+ clips",
        "strength": "Large scale multimodal clips with fine-grained intensity values",
        "primary_modality": "video/face | audio | text",
        "source": "Carnegie Mellon University",
        "download": "kaggle.com/cmu-mosei",
        "note": "Strong candidate for video/face + audio + text fusion validation once the video modality is chosen.",
    },
}

MODALITY_ORDER = ["voice", "face", "text"]


def compare_table() -> str:
    lines = [
        "# Modality Model Comparison (MELD-based Stress/Burnout Tracker)",
        "",
        "All three encoders are trained **separately first**, then projected to "
        "a common embedding dimension before the Cross-Modal Attention "
        "Transformer fuses them.",
        "",
        "## Comparison Table",
        "",
        "| Modality | Model | Fit | Critical limitation | Baseline accuracy | Best metric for the project |",
        "|---|---|---|---|---|---|",
    ]

    for mod in MODALITY_ORDER:
        rows = MODEL_REGISTRY[mod]
        for key, cfg in rows.items():
            lines.append(
                f"| {mod.capitalize()} | {cfg['name']} | {cfg['fit']} | "
                f"{cfg['limit']} | {cfg['baseline_acc']} | "
                f"{cfg['target_metric']} |"
            )
    lines += [
        "",
        "## One-line verdict",
        "",
        "For a **non-contact, real-time** stress tracker, this project should:",
        "",
        "1. Use **emotion2vec** as the primary *voice* stream.",
        "2. Keep **BERT/DeBERTa** for the *text log* stream (best chronic, "
        "cognitive-burnout signal).",
        "3. Use **MAE-DFER** for the *face* stream, with **EST** as the "
        "alternative and **Hybrid CNN + ViT** as the third candidate.",
        "4. Fuse all three through the **Cross-Modal Attention Transformer** "
        "so the wearable/classifier conflict is resolved by reliability "
        "weighting instead of majority vote.",
        "",
        "## Recommended encoder choice",
        "",
        "| Task | Primary encoder | Runner-up |",
        "|---|---|---|",
        "| Voice | emotion2vec | WavLM-Base / Wav2Vec2-Base |",
        "| Face | MAE-DFER | EST / Hybrid CNN + ViT |",
        "| Text | DeBERTa-v3-Base | RoBERTa-Base |",
        "",
        "## Datasets for context",
        "",
        "The datasets below are useful for **stress-specific "
        "validation** and for verifying that the Cross-Modal Transformer "
        "weights generalise beyond MELD. They are mapped to the modalities they "
        "best support: text, audio, and video/face.",
    ]

    for key, cfg in DATASETS.items():
        lines += [
            f"### {cfg['name']}",
            "",
            f"- Scale: {cfg['scale']}",
            f"- Strengths: {cfg['strength']}",
            f"- Ideal component focus: {cfg.get('primary_modality', cfg.get('ideal_component', 'n/a'))}",
            f"- Source: {cfg['source']}",
            f"- Download: {cfg['download']}",
            "",
        ]
    return "\n".join(lines)


def snapshot_path(table_path: str = "comparison_table.md") -> str:
    """Write `table_path` plus a machine-readable JSON snapshot next to it."""
    import json
    import pathlib

    paths = pathlib.Path(table_path)
    data = {
        "modalities": {m: {"models": list(MODEL_REGISTRY[m].keys())} for m in MODALITY_ORDER},
        "datasets": DATASETS,
        "modality_order": MODALITY_ORDER,
    }
    paths.write_text(compare_table(), encoding="utf-8")
    json_path = paths.with_suffix(".json")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(paths)


if __name__ == "__main__":
    print(compare_table())
