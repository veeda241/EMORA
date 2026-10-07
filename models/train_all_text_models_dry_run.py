"""Dry-run the GoEmotions text-model training loop.

This is a starting point for a smoke test. It verifies that the data loader,
tokenizer, multi-label model construction, and Hugging Face Trainer fit all the
way through one validation evaluation without doing any real optimization.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_all_text_models import main  # noqa: E402

if __name__ == "__main__":
    main(dry_run=True)
