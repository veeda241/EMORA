#!/usr/bin/env python3
"""Inspect the internal structure of the local CMU-MOSEI CSD files.

This script does NOT train anything and does NOT modify the raw files.
It reports sequence/clip/utterance structure, timestamps, feature dimensions,
sequence lengths, missing values, NaN/Inf, alignment relationships, and
official fold coverage.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# Project layout
SCRIPT = Path(__file__).resolve()
MODELS = SCRIPT.parent.parent
REPO_ROOT = MODELS.parent
ROOT = REPO_ROOT

# `data/` is at the repo root, e.g. `<repo>/data/raw/cmu_mosei/`.
DATA_RAW = REPO_ROOT / "data" / "raw" / "cmu_mosei"

# The CMU-MOSEI SDK source of truth for official folds.
SDK_FOLDS_PY = (
    ROOT
    / "data"
    / "external"
    / "CMU-MultimodalSDK-main"
    / "mmsdk"
    / "mmdatasdk"
    / "dataset"
    / "standard_datasets"
    / "CMU_MOSEI"
    / "cmu_mosei_std_folds.py"
)

# Robust lookup: the SDK folder may sit next to `models/` or next to the repo root.
def resolve_sdk_folds() -> Path:
    candidates = [
        ROOT / "data" / "external" / "CMU-MultimodalSDK-main" / "mmsdk" / "mmdatasdk" / "dataset" / "standard_datasets" / "CMU_MOSEI" / "cmu_mosei_std_folds.py",
        ROOT.parent / "CMU-MultimodalSDK-main" / "mmsdk" / "mmdatasdk" / "dataset" / "standard_datasets" / "CMU_MOSEI" / "cmu_mosei_std_folds.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find cmu_mosei_std_folds.py in any expected location. Tried:\n" + "\n".join(str(c) for c in candidates))

FILES = {
    "COVAREP": DATA_RAW / "CMU_MOSEI_COVAREP.csd",
    "Labels": DATA_RAW / "CMU_MOSEI_Labels.csd",
    "Words": DATA_RAW / "CMU_MOSEI_TimestampedWords.csd",
    "OpenFace2": DATA_RAW / "CMU_MOSEI_OpenFace2.csd",
    "Facet42": DATA_RAW / "CMU_MOSEI_VisualFacet42.csd",
}


def load_folds() -> tuple[set[str], set[str], set[str]]:
    ns: dict[str, object] = {}
    ns["__name__"] = "cmu_mosei_std_folds"
    with resolve_sdk_folds().open("r", encoding="utf-8") as fh:
        exec(fh.read(), ns)
    return (
        set(ns["standard_train_fold"]),
        set(ns["standard_valid_fold"]),
        set(ns["standard_test_fold"]),
    )


def top_group(f: h5py.File) -> str:
    """Return the single top-level data group name (e.g. 'COVAREP' or 'All Labels')."""
    keys = list(f.keys())
    if len(keys) != 1:
        raise ValueError(f"Unexpected root keys: {keys}")
    return keys[0]


def clip_list(f: h5py.File, top: str) -> list[str]:
    return list(f[f"{top}/data"].keys())


def sample_clip_info(
    f: h5py.File,
    top: str,
    clip: str,
) -> dict:
    data = f[f"{top}/data"][clip]
    keys = list(data.keys())
    features = data["features"]
    info: dict = {
        "clip": clip,
        "keys": keys,
        "features_shape": list(features.shape),
        "features_dtype": str(features.dtype),
        "has_intervals": "intervals" in data,
    }
    if info["has_intervals"]:
        info["intervals_shape"] = list(data["intervals"].shape)
        info["intervals_dtype"] = str(data["intervals"].dtype)
        iv = data["intervals"][0]
        info["interval_first"] = [float(x) for x in iv[:2]]
    if "metadata" in f[top]:
        meta = f[f"{top}/metadata"]
        if "dimension names" in meta:
            dn = meta["dimension names"][0]
            info["dimension_names"] = json.loads(str(dn, "utf-8"))
    return info


def stream_sequence_length_stats(
    f: h5py.File,
    top: str,
    clips: list[str],
    sample_limit: int = 600,
) -> dict:
    lengths: list[int] = []
    nan_count = 0
    inf_count = 0
    frames_seen = 0
    clips_seen = 0
    for clip in clips:
        feats = f[f"{top}/data/{clip}/features"][:]
        try:
            feats_num = np.asarray(feats, dtype=np.float64)
        except Exception:
            feats_num = feats
        lengths.append(int(feats.shape[0]))
        if np.issubdtype(np.asarray(feats_num).dtype, np.number):
            nan_count += int(np.isnan(feats_num).sum())
            inf_count += int(np.isinf(feats_num).sum())
        frames_seen += int(np.asarray(feats_num).size)
        clips_seen += 1
        if clips_seen >= sample_limit:
            break
    arr = np.array(lengths, dtype=np.int64)
    return {
        "clips_read_for_stats": clips_seen,
        "min_length": int(arr.min()),
        "max_length": int(arr.max()),
        "median_length": float(np.median(arr)),
        "mean_length": float(arr.mean()),
        "p95_length": int(np.percentile(arr, 95)),
        "nan_count_sampled": nan_count,
        "inf_count_sampled": inf_count,
        "frames_seen_sampled": frames_seen,
    }


def cross_clip_presence(
    sets: dict[str, set[str]],
    label_key: str = "Labels",
) -> dict:
    lset = sets[label_key]
    out: dict[str, dict[str, object]] = {}
    for key in sets:
        if key == label_key:
            continue
        cset = sets[key]
        out[key] = {
            "clips": len(cset),
            "clips_not_in_labels": len(cset - lset),
            "clips_in_labels": len(cset & lset),
            "labels_missing_this": len(lset - cset),
            "all_label_clips_have_this": bool(lset <= cset),
            "all_this_clips_are_labels": bool(cset <= lset),
        }
    return out


def utterance_modality_presence_sample(
    label_file: Path,
    modality_files: dict[str, Path],
    sample_limit: int = 25000,
) -> dict:
    """Estimate how many label utterances have overlapping modality frames.

    Because the files are large, this is sampled and used only for reporting.
    """
    train, valid, test = load_folds()
    all_clips = train | valid | test

    lf = h5py.File(label_file, "r")
    ltop = top_group(lf)
    ldata = lf[f"{ltop}/data"]

    loaders: dict[str, tuple[h5py.File, str, h5py.Group]] = {}
    for name, path in modality_files.items():
        mf = h5py.File(path, "r")
        mtop = top_group(mf)
        loaders[name] = (mf, mtop, mf[f"{mtop}/data"])

    counts: dict[str, Counter] = {k: Counter() for k in ("train", "valid", "test")}
    total = 0

    for clip in ldata:
        if clip not in all_clips:
            continue
        fold = ("train" if clip in train else "valid" if clip in valid else "test")
        lints = ldata[clip]["intervals"][:]
        batch = lints.shape[0]

        modality_intervals = {}
        for name, (mf, mtop, mdata) in loaders.items():
            if clip in mdata:
                modality_intervals[name] = mdata[clip]["intervals"][:]
            else:
                modality_intervals[name] = None

        for i in range(batch):
            lo, hi = float(lints[i, 0]), float(lints[i, 1])
            for name, mids in modality_intervals.items():
                if mids is None:
                    continue
                has = bool((mids[:, 1] > lo - 1e-6).any() and (mids[:, 0] < hi + 1e-6).any())
                if has:
                    counts[fold][name] += 1
            counts[fold]["utterances"] += 1
            total += 1
            if total >= sample_limit:
                break
        if total >= sample_limit:
            break

    for mf, _, _ in loaders.values():
        mf.close()
    lf.close()

    return {
        "sample_limit": total,
        "presence": {k: dict(v) for k, v in counts.items()},
    }


def main() -> None:
    train, valid, test = load_folds()
    all_clips = train | valid | test

    raw = {}
    for name, path in FILES.items():
        if not path.exists():
            raw[name] = {"error": "missing"}
            continue
        try:
            f = h5py.File(path, "r")
        except Exception as e:
            raw[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        try:
            top = top_group(f)
            clips = clip_list(f, top)
            info = sample_clip_info(f, top, clips[0])
            stats = stream_sequence_length_stats(f, top, clips)
            info.update(stats)
            info["clips"] = len(clips)
            raw[name] = info
        finally:
            f.close()

    sets: dict[str, set[str]] = {}
    for name, path in FILES.items():
        if "error" in raw.get(name, {}):
            sets[name] = set()
            continue
        f = h5py.File(path, "r")
        try:
            top = top_group(f)
            sets[name] = set(clip_list(f, top))
        finally:
            f.close()

    report: dict = {
        "files": {name: {"path": str(path)} for name, path in FILES.items()},
        "folds": {
            "standard_train_fold": len(train),
            "standard_valid_fold": len(valid),
            "standard_test_fold": len(test),
            "train_valid_overlap": len(train & valid),
            "train_test_overlap": len(train & test),
            "valid_test_overlap": len(valid & test),
            "clips_in_official_folds": len(all_clips),
        },
        "csd_structure": raw,
        "cross_clip_presence_vs_labels": cross_clip_presence(sets, "Labels"),
    }

    # Only sample utterance-level modality presence when the relevant modality files are readable.
    modality_files_for_presence = {
        k: v for k, v in FILES.items() if k != "Labels" and "error" not in report["csd_structure"].get(k, {})
    }
    if modality_files_for_presence:
        try:
            report["utterance_modality_presence_sample"] = utterance_modality_presence_sample(
                FILES["Labels"],
                modality_files_for_presence,
            )
        except Exception as exc:
            report["utterance_modality_presence_sample"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        report["utterance_modality_presence_sample"] = {
            "note": "skipped because no modality files were readable for the sampled alignment check",
        }

    out_dir = ROOT / "models" / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cmumosei_inspection_raw.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = out_dir / "cmumosei_inspection_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print("WROTE", json_path)
    print("WROTE", md_path)


def _render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# CMU-MOSEI Inspection Report")
    lines.append("")
    lines.append("Generated by `models/preprocessing/inspect_cmumosei.py`.")
    lines.append("This script does not modify any raw CSD file.")
    lines.append("")

    lines.append("## Files inspected")
    lines.append("")
    lines.append("| Name | Path |")
    lines.append("|---|---|")
    for name, entry in report["files"].items():
        lines.append(f"| {name} | `{entry['path']}` |")
    lines.append("")

    lines.append("## Official CMU-MOSEI folds (from SDK)")
    lines.append("")
    f = report["folds"]
    lines.append(f"- train clips: {f['standard_train_fold']}")
    lines.append(f"- validation clips: {f['standard_valid_fold']}")
    lines.append(f"- test clips: {f['standard_test_fold']}")
    lines.append(f"- clips appearing in any official fold: {f['clips_in_official_folds']}")
    lines.append("")
    lines.append(f"- train ∩ validation overlap: {f['train_valid_overlap']}")
    lines.append(f"- train ∩ test overlap: {f['train_test_overlap']}")
    lines.append(f"- validation ∩ test overlap: {f['valid_test_overlap']}")
    lines.append("")

    lines.append("## CSD internal structure")
    lines.append("")
    for name in FILES:
        info = report["csd_structure"].get(name, {})
        if "error" in info:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"`{FILES[name]}` could not be opened:")
            lines.append(f"- {info['error']}")
            lines.append("")
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- path: `{FILES[name]}`")
        lines.append(f"- top-level group: `{info.get('top_group', '')}`")
        lines.append(f"- total clips/sequences: {info.get('clips')}")
        lines.append(f"- sample clip: `{info.get('clip')}`")
        lines.append(f"- sample clip keys: {info.get('keys')}")
        lines.append(f"- feature dataset shape: {info.get('features_shape')}")
        lines.append(f"- feature dtype: {info.get('features_dtype')}")
        if "dimension_names" in info:
            dn = info["dimension_names"]
            lines.append(f"- dimension names ({len(dn)}): {dn}")
        if info.get("has_intervals"):
            lines.append(f"- intervals shape: {info.get('intervals_shape')}")
            lines.append(f"- intervals dtype: {info.get('intervals_dtype')}")
            lines.append(f"- first interval: {info.get('interval_first')}")
        else:
            lines.append("- intervals: not present")
        lines.append("")
        ls = info.get("min_length"), info.get("median_length"), info.get("max_length")
        lines.append("#### Sequence length stats (sampled)")
        lines.append("")
        lines.append(f"- clips read for stats: {info.get('clips_read_for_stats')}")
        lines.append(f"- min frames: {info.get('min_length')}")
        lines.append(f"- median frames: {info.get('median_length')}")
        lines.append(f"- max frames: {info.get('max_length')}")
        lines.append(f"- mean frames: {info.get('mean_length')}")
        lines.append(f"- 95th percentile frames: {info.get('p95_length')}")
        lines.append("")
        lines.append("#### Missing values (sampled)")
        lines.append("")
        lines.append(f"- NaN count: {info.get('nan_count_sampled')}")
        lines.append(f"- Inf count: {info.get('inf_count_sampled')}")
        lines.append(f"- frames seen: {info.get('frames_seen_sampled')}")
        lines.append("")

    lines.append("## Cross-clip presence vs Labels")
    lines.append("")
    lines.append("For each modality, how many clips are present, missing from Labels, etc.")
    lines.append("")
    lines.append("| Modality | total clips | clips not in Labels | clips in Labels | Labels missing this | all_label_clips_have_this |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in report["cross_clip_presence_vs_labels"].items():
        lines.append(
            f"| {name} | {r['clips']} | {r['clips_not_in_labels']} | "
            f"{r['clips_in_labels']} | {r['labels_missing_this']} | {r['all_label_clips_have_this']} |"
        )
    lines.append("")

    lines.append("## Utterance-level modality presence (sampled)")
    lines.append("")
    lines.append("This estimates whether each label utterance has overlapping modality frames.")
    lines.append("")
    lines.append(f"- sample limit: {report['utterance_modality_presence_sample']['sample_limit']} utterances")
    lines.append("")
    lines.append("| Split | utterances | Words | Facet42 | COVAREP |")
    lines.append("|---|---|---|---|---|")
    for split in ["train", "valid", "test"]:
        c = report["utterance_modality_presence_sample"]["presence"][split]
        lines.append(
            f"| {split} | {c.get('utterances',0)} | {c.get('Words',0)} | "
            f"{c.get('Facet42',0)} | {c.get('COVAREP',0)} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- COVAREP is an already-extracted acoustic feature representation, not raw audio.")
    lines.append("- OpenFace2 and Facet42 are already-extracted visual feature representations, not raw video frames.")
    lines.append("- CMU-MOSEI emotion labels are continuous scores, not single-class hard labels.")
    lines.append("- Any missing-modality or missing-file findings here are handled explicitly in the preprocessing pipeline.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
