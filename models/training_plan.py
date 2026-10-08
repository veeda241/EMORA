"""Training-plan generator for the MELD-based stress/burnout tracker.

Turns the model comparison grid into concrete PyTorch training steps per
modality. Each stream independently produces a fixed-dimension vector so the
Cross-Modal Attention Transformer can fuse them on equal terms.
"""

from __future__ import annotations

from models.compare import MODEL_REGISTRY, MODALITY_ORDER

PROJECT_DIM = 512  # shared embedding dimension fed into the Cross-Modal Transformer


def projection_block(in_dim: int, out_dim: int = PROJECT_DIM) -> str:
    return (
        f"    nn.Sequential(\n"
        f"        nn.Linear({in_dim}, {out_dim}),\n"
        f"        nn.LayerNorm({out_dim}),\n"
        f"        nn.GELU(),\n"
        f"        nn.Dropout({0.1}),\n"
        f"        nn.Linear({out_dim}, {out_dim}),\n"
        f"    ),"
    )


def modality_stream(modality: str, model_key: str) -> str:
    cfg = MODEL_REGISTRY[modality][model_key]
    name = cfg["name"]

    if modality == "text":
        in_dim = 768
        head = "    nn.Linear(768, 1)\n    nn.Sigmoid()\n"
        step = (
            "Fine-tune the encoder on MELD transcripts, take the [CLS]/mean "
            "pooled hidden state."
        )
    elif modality == "face":
        in_dim = 512
        head = "    nn.Linear(512, 1)\n    nn.Sigmoid()\n"
        step = (
            "Preprocess aligned face clips, then extract spatiotemporal "
            "features using MAE-DFER, EST, or the Hybrid CNN + ViT encoder."
        )
    else:
        in_dim = 768
        head = "    nn.Linear(768, 1)\n    nn.Sigmoid()\n"
        step = (
            "Extract fixed-size audio embeddings from emotion2vec, WavLM, or "
            "Wav2Vec2 on the audio tracks, then time-pool them before "
            "projection. emotion2vec is primary, WavLM is the comparison "
            "model, and Wav2Vec2 is the baseline."
        )

    return (
        f"def {modality}_stream({modality}_model: torch.nn.Module, "
        f"{modality}_dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.utils.data.DataLoader:\n"
        f'    """Train the {modality}-specific encoder and project to {PROJECT_DIM}-dim vector."""\n'
        f"    model = {modality}_model.to(device)\n"
        f"    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)\n"
        f"    criterion = torch.nn.BCELoss()\n"
        f"\n"
        f"    # Step 0 -> dataset/parsing (E.g. MELD CSV -> {modality} samples)\n"
        f"    # Step 1 -> {step}\n"
        f"    # Step 2 -> collect hidden-state vectors into a {in_dim}-dim tensor\n"
        f"    # Step 3 -> project to a uniform {PROJECT_DIM}-dim tensor\n"
        f"{projection_block(in_dim, PROJECT_DIM)}\n"
        f"    # Step 4 -> classification head\n"
        f"{head}"
        f"    # Step 5 -> self-supervised + supervised training loop, save best checkpoint\n"
        f"    return {modality}_model\n"
    )


def training_plan() -> str:
    out = [
        "# Multimodal Training Plan (MELD-based Stress/Burnout Tracker)",
        "",
        "Each modality is trained independently first. Every encoder emits a",
        f"fixed {PROJECT_DIM}-dimensional vector, so the Cross-Modal Attention",
        "Transformer can fuse them on equal terms without shape mismatches.",
        "",
        "## Prerequisites",
        "",
        "- Python 3.10+, PyTorch 2.x, Hugging Face `transformers`",
        "- GoEmotions (text), MELD (audio), and CMU-MOSEI (video/features)",
        f"- Project-wide projection target: `PROJECT_DIM = {PROJECT_DIM}`",
        "",
        "## Per-modality PyTorch streams",
        "",
    ]
    for mod in MODALITY_ORDER:
        cfg = MODEL_REGISTRY[mod]
        out.append(f"### {mod.capitalize()} Stream")
        for key, c in cfg.items():
            out.append(f"#### {c['name']} ({key})")
            out.append("```python")
            out.append(modality_stream(mod, key).rstrip())
            out.append("```")
            out.append("")
    out += [
        "## Step 6 -> Cross-Modal Attention Transformer",
        "",
        "After the three streams above return a {PROJECT_DIM}-dim vector each,",
        "stack them, normalize, and pass them through your multi-head cross-",
        "attention transformer encoder. The attention weights decode which",
        "modality is most reliable for the current sample (e.g. voice outranks",
        "face when the user is speaking, face outranks text when the user",
        "is silent).",
        "",
        "## Suggested baselines for the comparison table",
        "",
        "| Modality | Primary encoder | Runner-up |",
        "|---|---|---|",
        "| Voice | emotion2vec | WavLM-Base / Wav2Vec2-Base |",
        "| Face | MAE-DFER | EST / Hybrid CNN + ViT |",
        "| Text | DeBERTa-v3-Base | RoBERTa-Base |",
        "",
        "## Metrics to record across the table",
        "",
        "- **Voice**: inference latency + intensity tracking (Wav2Vec2)",
        "- **Face**: classification accuracy under angle/lighting change",
        "- **Text**: F1-score on conversational transcripts",
        "- **Fused**: final stress score accuracy after the Cross-Modal Transformer",
        "",
        "## Expected result",
        "",
        "Individually the three encoders land in the 74-92% range. Once fused",
        "through the Cross-Modal Attention Transformer the stress/burnout signal",
        "typically rises past the 85-90% accuracy ceiling that any single",
        "modality can reach alone.",
        "",
        "## Notes on the datasets",
        "",
        "- **MELD** is the main supervised source: natural multi-party",
        "  conversational flow that keeps BERT context retention sharp.",
        "- **CMU-MOSEI** adds 23,500+ clips with fine-grained intensity values",
        "  to tune the Cross-Modal Transformer weighting rules.",
        "- **GoEmotions** supplies the fine-grained text emotion labels; its",
        "  multilabel annotations must be handled explicitly by the text model.",
        "",
        "## Data preparation checklist",
        "",
        "1. Split MELD into train/val/test (speaker-independent).",
        "2. Align text transcripts, face crops and .wav tracks by utterance",
        "   timestamp so stream 1/2/3 see the same sample index.",
        "3. Freeze the audio/visual encoders initially; only train the",
        "   projection head + cross-modal attention first (avoids catastrophic",
        "   forgetting).",
        "4. Apply modality dropout (e.g. randomly mask the audio stream in 10%",
        "   of batches) so the Transformer learns to re-weight the remaining",
        "   modalities during real-world sensor dropouts.",
        "",
        "## Output artifacts",
        "",
        "- `models/comparison_table.md` - the model comparison table",
        "- `models/comparison_table.json` - machine-readable table snapshot",
        "- `models/{modality}_stream.py` - each PyTorch encoder stream",
        "- `models/cross_modal_attention_transformer.py` - fusion block",
    ]
    plan = "\n".join(out)
    return plan.replace("{PROJECT_DIM}", str(PROJECT_DIM))


if __name__ == "__main__":
    print(training_plan())
