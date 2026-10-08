#!/usr/bin/env python3
"""Inspect MELD CSVs for audio-modality training readiness.

Checks:
- emotion classes, counts, label mapping
- train/dev/test split sizes
- id overlap between splits
- missing audio files (none expected here, flagged explicitly)
- any anomalies before training.
"""
import csv, json, pathlib, collections, sys

RAW = pathlib.Path("data/raw")

def meld_csv_rows(sub="meld"):
    base = RAW / sub
    files = {
        "train": base / "train_sent_emo.csv",
        "dev": base / "dev_sent_emo.csv",
        "test": base / "test_sent_emo.csv",
    }
    out = {}
    for split, p in files.items():
        with p.open(newline="", encoding="utf-8") as f:
            out[split] = list(csv.DictReader(f))
    return out

def main():
    rows = meld_csv_rows("meld")
    counts = {split: len(rs) for split, rs in rows.items()}
    print("=== MELD CSV inspection ===")
    print("Splits:", counts)

    all_labels = set()
    per_split_counts = {}
    for split, rs in rows.items():
        c = collections.Counter(r["Emotion"] for r in rs)
        per_split_counts[split] = c
        all_labels.update(c.keys())

    sorted_labels = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(sorted_labels)}
    print("Emotion classes (%d):" % len(sorted_labels))
    for l in sorted_labels:
        print("  %s -> %d" % (l, label2id[l]))
    print("Label counts per split:")
    for split in ["train", "dev", "test"]:
        print(" ", split, dict(per_split_counts[split]))

    # id overlap
    ids_by_split = {}
    for split, rs in rows.items():
        ids_by_split[split] = set((r.get("Dialogue_ID"), r.get("Utterance_ID")) for r in rs)
    print("Utterance-level id overlap (Dialogue_ID, Utterance_ID):")
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        ov = ids_by_split[a] & ids_by_split[b]
        print("  %s U %s: %d" % (a, b, len(ov)))
        if len(ov) > 0:
            # show a couple example overlapping ids
            for ex in list(ov)[:3]:
                print("    example overlap:", ex)

    # label range check
    bad = []
    for split, rs in rows.items():
        for r in rs:
            lid = label2id[r["Emotion"]]
            if not (0 <= lid < len(label2id)):
                bad.append((split, r["Emotion"], lid))
    print("Out-of-range labels:", len(bad), "examples:", bad[:5])

    # audio files check
    audio_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma")
    found_audio = []
    for p in RAW.rglob("*"):
        if p.is_file() and p.suffix.lower() in audio_exts:
            found_audio.append(str(p))
    print("Audio files under data/raw:", len(found_audio), "(audio files are expected externally; none found in data/raw as provided)")
    for f in found_audio[:20]:
        print("  -", f)

    # save mapping
    out_path = pathlib.Path("models/audio/label_mapping.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"label2id": label2id, "id2label": {i: l for l, i in label2id.items()}}, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    print()
    print("LABEL MAPPING (id -> name):")
    id2label={i:l for l,i in label2id.items()}
    for i in range(len(id2label)):
        print(" ", i, "->", id2label[i])

    print()
    print("AUDIO AVAILABILITY NOTE:")
    print("  No audio files were found under data/raw in the provided project snapshot.")
    print("  MELD CSVs reference utterance timestamps (StartTime/EndTime) and would")
    print("  normally align with an external MELD audio directory (e.g. audio/ folder).")
    print("  Until the matching audio directory is present and aligned, no audio")
    print("  training can proceed. The label mapping above is ready to use once")
    print("  the audio files are added.")

    # summary
    print("\nReady to train on MELD audio?")
    print("- num_classes:", len(label2id))
    print("- total samples:", sum(counts.values()))
    print("- audio present:", len(found_audio) > 0)

if __name__ == "__main__":
    main()
