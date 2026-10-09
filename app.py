"""Gradio UI — Behaviour-Change Conversation Coach.

All three modalities (text, audio, video) work simultaneously in Live mode.
The Memory-Fusion Network (MFN) is available in Real MFN mode.

Run:
    python app.py          # http://127.0.0.1:7860
    AGENT_LLM=off python app.py   # deterministic templates, no LLM download
"""

from __future__ import annotations

import os
import traceback

import numpy as np
import torch
import gradio as gr

from src.config import CONFIG
from src.conversation.agent import MODEL_PATH, BehaviourChangeAgent
from src.pipeline.audio_pipeline import AudioPipeline
from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline
from src.pipeline.text_pipeline import TEXTCHECKPOINT_DIR, TextPipeline
from src.pipeline.video_pipeline import VideoPipeline, SEQ_LEN

# ── constants ───────────────────────────────────────────────────────────────
EMOTIONS     = CONFIG["label_names"]
NO_EMOTION   = "none (text only)"
REAL_MODE    = "Real MFN (CMU-MOSEI test sample)"
LIVE_MODE    = "Live input (text + audio + video)"
SIM_MODE     = "Simulated label (edge cases)"
TEST_NPZ     = f"{CONFIG['data_dir']}/test_data.npz"

_TEXT_WEIGHTS = any(
    (d / name).exists()
    for d in TEXTCHECKPOINT_DIR.glob("checkpoint-*")
    for name in ("model.safetensors", "pytorch_model.bin")
)
TEXT_LABEL = (
    f"GoEmotions {TEXTCHECKPOINT_DIR.name.removeprefix('runs_')} (fine-tuned)"
    if _TEXT_WEIGHTS
    else "GoEmotions pipeline (lexicon fallback — no fine-tuned text weights)"
)

LLM_CONFIG = None if os.environ.get("AGENT_LLM", "local") == "off" else {"backend": "local"}
LLM_LABEL = (
    f"local HF LLM ({(LLM_CONFIG or {}).get('model_id', 'Qwen/Qwen2.5-1.5B-Instruct').split('/')[-1]})"
    if LLM_CONFIG else "deterministic templates (AGENT_LLM=off)"
)

# ── lazy singletons ─────────────────────────────────────────────────────────
_MFN:        MultimodalEmotionPipeline | None = None
_TEST:       dict[str, np.ndarray] | None = None
_TEXT_PIPE:  TextPipeline   | None = None
_AUDIO_PIPE: AudioPipeline  | None = None
_VIDEO_PIPE: VideoPipeline  | None = None

_MFN_ERROR  = ""
_AUDIO_ERROR = ""
_VIDEO_ERROR = ""


def get_mfn() -> MultimodalEmotionPipeline | None:
    global _MFN, _MFN_ERROR
    if _MFN is not None:
        return _MFN
    if _MFN_ERROR:
        return None
    try:
        model = MultimodalEmotionPipeline(CONFIG)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
        model.eval()
        _MFN = model
    except Exception as exc:
        _MFN_ERROR = f"MFN unavailable: {exc}"
    return _MFN


def get_test_data() -> dict[str, np.ndarray]:
    global _TEST
    if _TEST is None:
        _TEST = dict(np.load(TEST_NPZ))
    return _TEST


def get_text_pipe() -> TextPipeline:
    global _TEXT_PIPE
    if _TEXT_PIPE is None:
        _TEXT_PIPE = TextPipeline()
    return _TEXT_PIPE


def get_audio_pipe() -> AudioPipeline | None:
    global _AUDIO_PIPE, _AUDIO_ERROR
    if _AUDIO_PIPE is not None:
        return _AUDIO_PIPE
    if _AUDIO_ERROR:
        return None
    try:
        _AUDIO_PIPE = AudioPipeline("emotion2vec")
    except Exception as exc:
        _AUDIO_ERROR = f"Audio pipeline unavailable: {exc}"
    return _AUDIO_PIPE


def get_video_pipe() -> VideoPipeline | None:
    global _VIDEO_PIPE, _VIDEO_ERROR
    if _VIDEO_PIPE is not None:
        return _VIDEO_PIPE
    if _VIDEO_ERROR:
        return None
    try:
        _VIDEO_PIPE = VideoPipeline()
    except Exception as exc:
        _VIDEO_ERROR = f"Video pipeline unavailable: {exc}"
    return _VIDEO_PIPE


def ensure_agent(agent, goal: str) -> BehaviourChangeAgent:
    if agent is None:
        agent = BehaviourChangeAgent(
            goal=goal.strip() or "general wellbeing",
            device="cpu",
            llm_config=LLM_CONFIG,
        )
    return agent


def simulate_vector(label: str, confidence: float) -> np.ndarray:
    vec = np.full(len(EMOTIONS), (1.0 - confidence) / max(len(EMOTIONS) - 1, 1))
    vec[EMOTIONS.index(label)] = confidence
    return vec


def sample_features(idx: int):
    d = get_test_data()
    i = max(0, min(int(idx), len(d["labels"]) - 1))
    return d["text"][i], d["audio"][i], d["video"][i]


def preview_sample(idx: int) -> str:
    mfn = get_mfn()
    if mfn is None:
        return f"⚠️ MFN not loaded. {_MFN_ERROR}"
    d = get_test_data()
    i = max(0, min(int(idx), len(d["labels"]) - 1))
    t, a, v = sample_features(i)
    true_label = EMOTIONS[int(d["labels"][i])]
    out = mfn.predict_emotion(
        torch.tensor(t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(a, dtype=torch.float32).unsqueeze(0),
        torch.tensor(v, dtype=torch.float32).unsqueeze(0),
    )
    probs = out["probs"][0]
    order = torch.argsort(probs, descending=True)
    top = ", ".join(f"{EMOTIONS[j]} {probs[j]:.2f}" for j in order[:3])
    return (
        f"**Text ✓ Audio ✓ Video ✓** — GloVe / COVAREP / FACET\n\n"
        f"True label: **{true_label}** | MFN prediction: **{out['label'][0]}** "
        f"(conf {float(out['confidence'][0]):.2f})\n\n"
        f"Top-3: {top}"
    )


def random_index() -> int:
    return int(np.random.randint(0, len(get_test_data()["labels"])))


# ── status helpers ──────────────────────────────────────────────────────────

def pipeline_status() -> str:
    lines = ["### Pipeline status"]
    # Text
    try:
        tp = get_text_pipe()
        lines.append(f"- **Text**: ✅ {tp.backend_name()}")
    except Exception as e:
        lines.append(f"- **Text**: ❌ {e}")
    # Audio
    ap = get_audio_pipe()
    if ap:
        lines.append("- **Audio (emotion2vec)**: ✅ loaded")
    else:
        lines.append(f"- **Audio**: ❌ {_AUDIO_ERROR or 'not loaded'}")
    # Video
    vp = get_video_pipe()
    if vp:
        lines.append(f"- **Video (hybrid-cnn-vit)**: ✅ loaded")
    else:
        lines.append(f"- **Video**: ❌ {_VIDEO_ERROR or 'not loaded'}")
    # MFN
    mfn = get_mfn()
    if mfn:
        lines.append("- **MFN (Memory Fusion)**: ✅ loaded")
    else:
        lines.append(f"- **MFN**: ❌ {_MFN_ERROR or 'not loaded'}")
    return "\n".join(lines)


# ── core respond function ───────────────────────────────────────────────────

def respond(message, history, agent, mode, sample_idx, sim_label, sim_conf,
            goal, audio, webcam_frame):
    if not message or not message.strip():
        return history, agent, gr.update(), gr.update(), ""

    agent = ensure_agent(agent, goal)
    preds   = None
    aligned = None
    source  = "none"
    warnings: list[str] = []

    # ── Real MFN mode ──
    if mode == REAL_MODE:
        mfn = get_mfn()
        if mfn is None:
            warnings.append(f"⚠️ {_MFN_ERROR} — falling back to text-only.")
            preds = {"text": np.asarray(get_text_pipe().predict(message.strip())["probs_array"])}
            source = "text-only fallback (MFN unavailable)"
        else:
            aligned = sample_features(sample_idx)
            source = f"MFN test sample #{int(sample_idx)}"

    # ── Live input mode ──
    elif mode == LIVE_MODE:
        preds = {}
        src_parts = []

        # Text — always runs
        tp = get_text_pipe()
        preds["text"] = np.asarray(tp.predict(message.strip())["probs_array"])
        src_parts.append(f"text:{tp.backend_name()}")

        # Audio — runs when the user has recorded/uploaded audio
        if audio is not None:
            try:
                sr, raw = audio
                raw = np.asarray(raw)
                # Convert int PCM to float32
                if raw.dtype.kind in ("i", "u"):
                    raw = raw.astype(np.float32) / 32768.0
                wav = raw.astype(np.float32).squeeze()
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)           # stereo → mono
                if wav.size > 0:
                    ap = get_audio_pipe()
                    if ap is not None:
                        result = ap.predict(wav, int(sr))
                        preds["audio"] = np.asarray(result["probs_array"])
                        src_parts.append(f"audio:emotion2vec({result['label']} {result['confidence']:.2f})")
                    else:
                        warnings.append(f"⚠️ {_AUDIO_ERROR}")
            except Exception as exc:
                warnings.append(f"⚠️ Audio processing error: {exc}")
        else:
            warnings.append("ℹ️ No audio provided — record or upload a clip in the Audio panel.")

        # Video — live webcam frame processed through MediaPipe + model
        if webcam_frame is not None:
            try:
                import cv2
                vp = get_video_pipe()
                if vp is not None:
                    # Gradio Image gives RGB numpy array
                    frame_rgb = np.asarray(webcam_frame)
                    if frame_rgb.dtype != np.uint8:
                        frame_rgb = (frame_rgb * 255).clip(0, 255).astype(np.uint8)
                    result = vp.predict_webcam(frame_rgb)
                    if result is not None and result.get("buffer_ready"):
                        preds["video"] = result["probs"]
                        buf = result["buffer_frames"]
                        src_parts.append(
                            f"video:live(AU+pose {result['label']} "
                            f"{result['confidence']:.2f} buf={buf}/{SEQ_LEN})"
                        )
                    elif result is not None:
                        buf = result.get("buffer_frames", 0)
                        warnings.append(
                            f"ℹ️ Video buffer filling ({buf}/{SEQ_LEN} frames) — keep face visible."
                        )
                    else:
                        warnings.append("ℹ️ No face detected in webcam frame.")
                else:
                    warnings.append(f"⚠️ {_VIDEO_ERROR}")
            except Exception as exc:
                warnings.append(f"⚠️ Video processing error: {exc}")

        source = "live — " + "; ".join(src_parts) if src_parts else "live — text only"

    # ── Simulated label mode ──
    elif mode == SIM_MODE:
        preds = {"text": simulate_vector(sim_label, float(sim_conf))}
        source = f"simulated — {sim_label} @ {sim_conf:.0%}"

    # ── No emotion mode ──
    else:
        source = NO_EMOTION

    # ── Run agent ──
    try:
        out = agent.respond(
            message.strip(),
            modality_predictions=preds,
            aligned_sequences=aligned,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        return (
            history + [
                [message.strip(), f"⚠️ Agent error: {exc}\n\n{tb}"],
            ],
            agent, None, None, "",
        )

    # ── Build trace ──
    emo = out["emotion"]
    detection = None
    if emo:
        used  = emo.get("modalities_used", [])
        marks = " ".join(
            f"{m.capitalize()} {'✓' if m in used else '✗'}"
            for m in ("text", "audio", "video")
        )
        detection = {
            "label":       emo["label"],
            "confidence":  round(emo["confidence"], 4),
            "modalities":  marks,
            "source":      source,
        }
    if warnings:
        detection = detection or {}
        detection["warnings"] = warnings

    trace = {
        "turn":                    agent.memory.turn_count // 2,
        "1_emotion_detection":     detection,
        "2_emotional_state":       out["emotional_state"],
        "3_response_strategy":     out["strategy"],
        "4_reply":                 out["reply"],
    }

    new_history = history + [
        [message.strip(), out["reply"]],
    ]
    return new_history, agent, trace, agent.memory.summary(), ""


def reset_conversation(goal, _history, _agent):
    vp = get_video_pipe()
    if vp is not None:
        vp.clear_buffer()
    agent = BehaviourChangeAgent(
        goal=goal.strip() or "general wellbeing",
        device="cpu",
        llm_config=LLM_CONFIG,
    )
    return [], agent, None, agent.memory.summary()


def toggle_mode(mode):
    return (
        gr.update(visible=(mode == REAL_MODE)),
        gr.update(visible=(mode == SIM_MODE)),
        gr.update(visible=(mode == LIVE_MODE)),
    )


# ── Build UI ────────────────────────────────────────────────────────────────

# Load test data eagerly so slider max is correct at build time.
_n_test = len(get_test_data()["labels"])

with gr.Blocks(title="Behaviour-Change Coach") as demo:
    gr.Markdown(
        "## 🧠 Behaviour-Change Coach — Multimodal Pipeline\n"
        f"Emotion detection → emotional state → memory → strategy → reply  |  LLM: {LLM_LABEL}"
    )
    agent_state = gr.State(None)

    with gr.Row():
        # ── Left column: chat ─────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=420)
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message and press Enter or click Send...",
                lines=2,
            )
            with gr.Row():
                send_btn  = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("🔄 Reset conversation")

            gr.Markdown(pipeline_status())

        # ── Right column: controls ────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 🎛️ Emotion detection mode")
            mode = gr.Radio(
                [REAL_MODE, LIVE_MODE, SIM_MODE, NO_EMOTION],
                value=LIVE_MODE,
                label="Source",
            )

            # Real MFN panel
            with gr.Group(visible=False) as real_group:
                sample_idx = gr.Slider(
                    0, _n_test - 1, value=0, step=1, precision=0,
                    label="CMU-MOSEI test utterance index",
                )
                random_btn     = gr.Button("🎲 Random sample")
                sample_preview = gr.Markdown(
                    "Switch to Real MFN mode to preview a sample."
                )

            # Simulated label panel
            with gr.Group(visible=False) as sim_group:
                sim_label = gr.Radio(EMOTIONS, value="sad", label="Simulated emotion")
                sim_conf  = gr.Slider(0.05, 1.0, value=0.8, step=0.05, label="Confidence")

            # Live input panel (default)
            with gr.Group(visible=True) as live_group:
                gr.Markdown(
                    "**Text** is always processed. "
                    "**Record or upload audio** to activate the audio emotion model. "
                    "**Enable video** to add a precomputed FACET segment."
                )

                # ── Audio ──
                with gr.Accordion("🎤 Audio", open=True):
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="Record or upload audio (WAV / MP3)",
                        format="wav",
                    )
                    gr.Markdown(
                        "_Record: click the mic icon → speak → click Stop. "
                        "The clip is sent automatically when you click **Send**._"
                    )

                # ── Video ──
                with gr.Accordion("📹 Video (live webcam)", open=False):
                    webcam_in = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam feed — point your face at the camera",
                        streaming=True,
                    )
                    video_status = gr.Markdown(
                        "_Enable the webcam above. Emotion is predicted from "
                        f"the last {SEQ_LEN} frames captured while you speak. "
                        "MediaPipe FaceMesh extracts Action Units + head pose in real-time._"
                    )

            goal_box = gr.Textbox(
                label="Coaching goal (used on Reset)",
                value="reduce daily stress",
            )

            # ── Pipeline trace ────────────────────────────────────────────
            gr.Markdown("### 🔍 Pipeline trace (last turn)")
            trace_json  = gr.JSON(label="Emotion → State → Strategy → Reply")
            memory_json = gr.JSON(label="Conversation memory")

    # ── Wire up events ────────────────────────────────────────────────────
    inputs = [
        msg, chatbot, agent_state, mode, sample_idx, sim_label, sim_conf,
        goal_box, audio_in, webcam_in,
    ]
    outputs = [chatbot, agent_state, trace_json, memory_json, msg]

    msg.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)
    reset_btn.click(
        reset_conversation,
        [goal_box, chatbot, agent_state],
        [chatbot, agent_state, trace_json, memory_json],
    )

    # Mode toggle
    mode.change(toggle_mode, [mode], [real_group, sim_group, live_group])

    # Real MFN preview
    sample_idx.change(preview_sample, [sample_idx], [sample_preview])
    random_btn.click(random_index, outputs=[sample_idx])

    # Init memory display on load
    demo.load(
        lambda: (
            None,
            BehaviourChangeAgent(
                goal="reduce daily stress",
                device="cpu",
                llm_config=LLM_CONFIG,
            ).memory.summary(),
        ),
        outputs=[trace_json, memory_json],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )
