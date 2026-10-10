"""Generate the frozen-model prediction table for decision-level fusion.

For every MELD utterance that has a per-utterance mp4, run the three FROZEN
models on the SAME sample and record their 6-class probability vectors:

    text  : TextPipeline (GoEmotions RoBERTa -> 6 shared classes)
    audio : AudioPipeline (emotion2vec head) on 16 kHz mono wav pulled from the mp4
    video : VideoPipeline (hybrid-cnn-vit) on FACET features from 16 sampled frames

The video path deliberately replicates the LIVE app path exactly: the FACET
extractor is created with apply_norm=True and the (16,35) sequence is fed to the
model directly (pool_frames + model), NOT through VideoPipeline.predict() which
would normalise a second time. This keeps the fusion's training distribution
identical to what the live agent will produce.

Output CSV columns: label, split, meld_split, dialogue_id, utterance_id,
then text_p_<c>, audio_p_<c>, video_p_<c> for each shared class <c>.

Rows where any modality is unavailable (no face detected / no audio stream) are
SKIPPED, never fabricated. Supports resume: re-running skips already-written
(dialogue_id, utterance_id, meld_split) keys.

Run:  python scripts/build_fusion_predictions.py
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.label_mapping import SHARED_EMOTION_CLASSES as CLASSES  # noqa: E402
from src.pipeline.text_pipeline import TextPipeline  # noqa: E402
from src.pipeline.audio_pipeline import AudioPipeline  # noqa: E402
from src.pipeline.video_pipeline import VideoPipeline, pool_frames, SEQ_LEN  # noqa: E402

MELD_ROOT = Path("D:/meld_raw/MELD.Raw")
MELD_CSV_DIR = ROOT / "data" / "processed" / "text" / "meld"

# MELD 7-class -> shared 6-class. 'neutral' has no shared counterpart -> dropped.
MELD2SHARED = {
    "anger": "angry", "disgust": "disgust", "fear": "fearful",
    "joy": "happy", "sadness": "sad", "surprise": "surprised",
}

# meld split key -> (processed csv, mp4 subdir, role in the fusion experiment)
SPLIT_CFG = {
    "dev":  {"csv": "meld_validation.csv", "dir": "dev_splits_complete",            "role": "train"},
    "test": {"csv": "meld_test.csv",       "dir": "output_repeated_splits_test",    "role": "test"},
}

MODALITIES = ("text", "audio", "video")

# Cap per-clip audio fed to emotion2vec (its attention memory grows ~quadratically).
MAX_AUDIO_SEC = 15.0


def feature_cols() -> list[str]:
    return [f"{m}_p_{c}" for m in MODALITIES for c in CLASSES]


def read_all_frames(mp4: Path):
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def sample_uniform(frames: list, n: int) -> list:
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def extract_wav_16k(mp4: Path, ffmpeg: str):
    """Pull a 16 kHz mono float waveform out of the mp4. Returns None if no audio."""
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = [ffmpeg, "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000",
               "-f", "wav", tmp.name]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or os.path.getsize(tmp.name) < 1000:
            return None
        wav, sr = sf.read(tmp.name, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if wav.size < 1600:  # <0.1 s — too short to be meaningful
            return None
        # emotion2vec self-attention memory grows ~quadratically with length;
        # very long MELD clips OOM the CPU. Cap to the first MAX_AUDIO_SEC.
        max_samples = int(MAX_AUDIO_SEC * sr)
        if wav.size > max_samples:
            wav = wav[:max_samples]
        return wav, int(sr)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def video_probs_live_path(vp: VideoPipeline, extractor, mp4: Path):
    """Replicate the live app video inference on a per-utterance clip."""
    import cv2
    frames = read_all_frames(mp4)
    if not frames:
        return None
    feats = []
    for fr in frames:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        f = extractor.extract(rgb)
        if f is not None:
            feats.append(f)
    if len(feats) == 0:
        return None  # no face in any frame -> video unavailable for this sample
    arr = np.stack(feats, axis=0)               # (m,35) already normalised by extractor
    seq = pool_frames(arr, SEQ_LEN)             # (16,35)
    x = torch.tensor(seq, dtype=torch.float32, device=vp.device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = vp.model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return probs


def load_done_keys(out_csv: Path) -> set:
    done = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((row["meld_split"], row["dialogue_id"], row["utterance_id"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "fusion_predictions.csv")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap utterances per (split,label) for balance; 0 = no cap")
    ap.add_argument("--limit", type=int, default=0, help="hard cap total rows per split (debug)")
    args = ap.parse_args()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    print("Loading frozen models (audio model may download emotion2vec on first use)...", flush=True)
    tp = TextPipeline()
    ap_ = AudioPipeline("emotion2vec")
    vp = VideoPipeline()
    extractor = vp._get_extractor()   # apply_norm=True, matches the live app
    print("Models ready. text=%s audio=emotion2vec video=%s" % (tp.backend_name(), vp.model_key), flush=True)

    header = ["label", "split", "meld_split", "dialogue_id", "utterance_id", *feature_cols()]
    write_header = not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.out)
    if done:
        print(f"Resume: {len(done)} rows already written, will skip them.", flush=True)

    import pandas as pd
    fh = args.out.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(header)
        fh.flush()

    stats = {"written": 0, "skip_done": 0, "no_mp4": 0, "no_video": 0, "no_audio": 0, "dropped_neutral": 0}
    t0 = time.time()

    for meld_split in args.splits.split(","):
        cfg = SPLIT_CFG[meld_split.strip()]
        mp4_dir = MELD_ROOT / cfg["dir"]
        df = pd.read_csv(MELD_CSV_DIR / cfg["csv"])
        role = cfg["role"]
        per_class: dict[str, int] = {}
        n_split = 0

        for _, row in df.iterrows():
            meld_label = str(row["label"]).strip().lower()
            if meld_label == "neutral":
                stats["dropped_neutral"] += 1
                continue
            shared = MELD2SHARED.get(meld_label)
            if shared is None:
                continue
            if args.max_per_class:
                if per_class.get(shared, 0) >= args.max_per_class:
                    continue
            dia, utt = int(row["dialogue_id"]), int(row["utterance_id"])
            key = (meld_split, str(dia), str(utt))
            if key in done:
                stats["skip_done"] += 1
                continue

            mp4 = mp4_dir / f"dia{dia}_utt{utt}.mp4"
            if not mp4.exists():
                stats["no_mp4"] += 1
                continue

            text = str(row["text"]).strip()
            t_probs = np.asarray(tp.predict(text)["probs_array"], dtype=np.float64)

            wav = extract_wav_16k(mp4, ffmpeg)
            if wav is None:
                stats["no_audio"] += 1
                continue
            a_probs = np.asarray(ap_.predict(wav[0], wav[1])["probs_array"], dtype=np.float64)

            v_probs = video_probs_live_path(vp, extractor, mp4)
            if v_probs is None:
                stats["no_video"] += 1
                continue

            probs = {"text": t_probs, "audio": a_probs, "video": v_probs}
            out_row = [shared, role, meld_split, dia, utt]
            for m in MODALITIES:
                out_row += [f"{probs[m][i]:.6f}" for i in range(len(CLASSES))]
            writer.writerow(out_row)
            done.add(key)
            stats["written"] += 1
            per_class[shared] = per_class.get(shared, 0) + 1
            n_split += 1

            if stats["written"] % 25 == 0:
                fh.flush()
                el = time.time() - t0
                rate = stats["written"] / el if el else 0
                print(f"[{meld_split}] written={stats['written']} rate={rate:.2f}/s "
                      f"elapsed={el/60:.1f}m drops(audio/video/mp4)="
                      f"{stats['no_audio']}/{stats['no_video']}/{stats['no_mp4']}", flush=True)

            if args.limit and n_split >= args.limit:
                print(f"[{meld_split}] hit --limit {args.limit}", flush=True)
                break

    fh.flush()
    fh.close()
    print("\nDONE", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  total time: {(time.time()-t0)/60:.1f} min")
    print(f"  wrote -> {args.out}")


if __name__ == "__main__":
    main()
