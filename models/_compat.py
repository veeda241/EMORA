"""Minor transformers compatibility shim.

Transformers < 4.46 does not expose ``TrainingArguments.dry_run``, but the
text-model trainer's dry-run path relies on it. Add the attribute at runtime so
the same trainer code can be smoke-tested without a real epoch.
"""

from __future__ import annotations

import inspect
from transformers import TrainingArguments

if not hasattr(TrainingArguments, "dry_run"):
    def _get_dry_run(self):
        return getattr(self, "_dry_run_override", False)

    def _set_dry_run(self, value):
        self._dry_run_override = bool(value)
        return value

    TrainingArguments.dry_run = property(_get_dry_run)
    TrainingArguments.set_dry_run = _set_dry_run
    TrainingArguments.dry_run = True  # type: ignore[assignment]
