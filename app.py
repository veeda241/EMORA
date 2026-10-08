"""Gradio test UI for the end-to-end behaviour-change conversation pipeline.

Emotion sources, selectable per turn:
    1. Real MFN inference on a real CMU-MOSEI test utterance
       (GloVe 300-d + COVAREP 74-d + FACET 35-d -> trained MFN -> 6-class softmax).
    2. Live input — the per-modality models on what you type and record:
       text via the fine-tuned GoEmotions text model (lexicon fallback only
       if no weights are on disk), audio via emotion2vec (the winning SER
       model, 92.7% test acc) on mic/upload, video only as a clearly-labelled
       precomputed FACET segment (no live FACET extractor exists).
    3. Simulated label (explicitly labelled; for deterministic edge cases).
    4. none — no emotion signal, tests the text-only fallback.

Everything downstream of detection — memory, behavioural trend, emotional
state, response strategy, reply — is the real conversation system.

Run:  python app.py  ->  http://127.0.0.1:7860
"""

from __future__ import annotations

import os

import numpy as np
import torch
import gradio as gr

from src.config import CONFIG
from src.conversation.agent import MODEL_PATH, BehaviourChangeAgent
from src.pipeline.audio_pipeline import AudioPipeline
from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline
from src.pipeline.text_pipeline import TEXTCHECKPOINT_DIR, TextPipeline
from src.pipeline.video_pipeline import VideoPipeline

EMOTIONS = CONFIG["label_names"]
NO_EMOTION = "none (text only)"
REAL_MODE = "Real MFN (CMU-MOSEI test sample)"
LIVE_MODE = "Live input (text + mic)"
SIM_MODE = "Simulated label (edge cases)"
TEST_NPZ = f"{CONFIG['data_dir']}/test_data.npz"

# Cheap file-existence probe (no model load) so the UI describes the text
# backend truthfully whether or not fine-tuned weights are on disk.
_TEXT_WEIGHTS = any(
    (d / name).exists()
    for d in TEXTCHECKPOINT_DIR.glob("checkpoint-*")
    for name in ("model.safetensors", "pytorch_model.bin")
)
TEXT_LABEL = (
    f"GoEmotions {TEXTCHECKPOINT_DIR.name.removeprefix('runs_')} (fine-tuned)"
    if _TEXT_WEIGHTS
    else "GoEmotions pipeline (lexicon fallback — no fine-tuned text weights on disk)"
)

# AGENT_LLM=off falls back to the deterministic template generator.
LLM_CONFIG = None if os.environ.get("AGENT_LLM", "local") == "off" else {"backend": "local"}
LLM_LABEL = (
    f"local Hugging Face LLM ({LLM_CONFIG.get('model_id', 'Qwen/Qwen2.5-1.5B-Instruct').split('/')[-1]})"
    if LLM_CONFIG else "deterministic templates (AGENT_LLM=off)"
)

_MFN: MultimodalEmotionPipeline | None = None
_TEST: dict[str, np.ndarray] | None = None
_TEXT_PIPE: TextPipeline | None = None
_AUDIO_PIPE: AudioPipeline | None = None
_VIDEO_PIPE: VideoPipeline | None = None


def get_mfn() -> MultimodalEmotionPipeline:
    global _MFN
    if _MFN is None:
        model = MultimodalEmotionPipeline(CONFIG)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
        model.eval()
        _MFN = model
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


def get_audio_pipe() -> AudioPipeline:
    global _AUDIO_PIPE
    if _AUDIO_PIPE is None:
        _AUDIO_PIPE = AudioPipeline("emotion2vec")
    return _AUDIO_PIPE


def get_video_pipe() -> VideoPipeline:
    global _VIDEO_PIPE
    if _VIDEO_PIPE is None:
        _VIDEO_PIPE = VideoPipeline()
    return _VIDEO_PIPE


def ensure_agent(agent: BehaviourChangeAgent | None, goal: str) -> BehaviourChangeAgent:
    if agent is None:
        agent = BehaviourChangeAgent(goal=goal.strip() or "general wellbeing", device="cpu", llm_config=LLM_CONFIG)
    return agent


def simulate_vector(label: str, confidence: float) -> np.ndarray:
    vec = np.full(len(EMOTIONS), (1.0 - confidence) / (len(EMOTIONS) - 1))
    vec[EMOTIONS.index(label)] = confidence
    return vec


def sample_features(idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, D) per modality — agent._mfn_predict adds the batch dim itself."""
    d = get_test_data()
    i = max(0, min(int(idx), len(d["labels"]) - 1))
    return d["text"][i], d["audio"][i], d["video"][i]


def preview_sample(idx: int) -> str:
    """Run the real MFN on the selected test utterance and format the result."""
    d = get_test_data()
    i = max(0, min(int(idx), len(d["labels"]) - 1))
    t, a, v = sample_features(i)
    true_label = EMOTIONS[int(d["labels"][i])]
    out = get_mfn().predict_emotion(
        torch.tensor(t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(a, dtype=torch.float32).unsqueeze(0),
        torch.tensor(v, dtype=torch.float32).unsqueeze(0),
    )
    probs = out["probs"][0]
    order = torch.argsort(probs, descending=True)
    top = ", ".join(f"{EMOTIONS[j]} {probs[j]:.2f}" for j in order[:3])
    return (
        f"**Text ✓ &nbsp; Audio ✓ &nbsp; Video ✓** (GloVe / COVAREP / FACET)\n\n"
        f"true label: **{true_label}** &nbsp;|&nbsp; MFN: **{out['label'][0]}** "
        f"(confidence {float(out['confidence'][0]):.2f})\n\n"
        f"top-3: {top}"
    )


def random_index() -> int:
    return int(np.random.randint(0, len(get_test_data()["labels"])))


def respond(message, history, agent, mode, sample_idx, sim_label, sim_conf, goal,
            audio, live_video, video_idx):
    if not message or not message.strip():
        return history, agent, gr.update(), gr.update(), ""
    agent = ensure_agent(agent, goal)
    source = None
    aligned, preds = None, None
    if mode == REAL_MODE:
        aligned = sample_features(sample_idx)
        source = f"MFN test sample #{int(sample_idx)}"
    elif mode == LIVE_MODE:
        preds, src_parts = {}, []
        tp = get_text_pipe()
        preds["text"] = np.asarray(tp.predict(message.strip())["probs_array"])
        src_parts.append(f"text: {tp.backend_name()}")
        if audio is not None:
            sr, raw = audio
            raw = np.asarray(raw)
            if raw.dtype.kind in "iu":
                raw = raw.astype(np.float32) / 32768.0
            wav = raw.astype(np.float32).squeeze()
            if wav.ndim == 1 and wav.size > 0:
                ap = get_audio_pipe()
                preds["audio"] = np.asarray(ap.predict(wav, int(sr))["probs_array"])
                src_parts.append("audio: emotion2vec (mic/upload)")
        if live_video:
            vp = get_video_pipe()
            d = get_test_data()
            seg = d["video"][max(0, min(int(video_idx), len(d["labels"]) - 1))]
            preds["video"] = vp.predict(seg)["probs"]
            src_parts.append(f"video: {vp.model_key} (precomputed FACET #{int(video_idx)}, not live)")
        source = "live — " + "; ".join(src_parts)
    elif mode == SIM_MODE:
        preds = {"text": simulate_vector(sim_label, float(sim_conf))}
        source = "simulated label"

    out = agent.respond(message.strip(), modality_predictions=preds, aligned_sequences=aligned)

    emo = out["emotion"]
    detection = None
    if emo:
        used = emo.get("modalities_used", [])
        marks = " ".join(f"{m.capitalize()} {'✓' if m in used else '✗'}" for m in ("text", "audio", "video"))
        detection = {
            "label": emo["label"],
            "confidence": round(emo["confidence"], 4),
            "modalities": marks,
            "source": source,
        }
    trace = {
        "turn": agent.memory.turn_count // 2,
        "1_emotion_detection": detection,
        "2_emotional_state": out["emotional_state"],
        "3_response_strategy": out["strategy"],
        "4_reply": out["reply"],
    }
    return history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": out["reply"]},
    ], agent, trace, agent.memory.summary(), ""


def reset_conversation(goal, _history, _agent):
    agent = BehaviourChangeAgent(goal=goal.strip() or "general wellbeing", device="cpu", llm_config=LLM_CONFIG)
    return [], agent, None, agent.memory.summary()


def toggle_mode(mode):
    return (
        gr.update(visible=mode == REAL_MODE),
        gr.update(visible=mode == SIM_MODE),
        gr.update(visible=mode == LIVE_MODE),
    )


with gr.Blocks(title="Behaviour-Change Coach — Pipeline Test UI") as demo:
    gr.Markdown(
        "## Behaviour-Change Coach — pipeline test UI\n"
        "Chain under test: **emotion detection → emotional state → memory / behavioural "
        "trend → response strategy → reply**. Default detection runs the **trained MFN on a "
        "real CMU-MOSEI test utterance** (GloVe + COVAREP + FACET); **Live input** runs the "
        "per-modality models on what you type and record (video limited to labelled "
        "precomputed FACET); your typed message always drives the conversation. "
        "Reply generation: " + LLM_LABEL + "."
    )
    agent_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            msg = gr.Textbox(label="Your message", placeholder="Type a message and press Enter...")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset conversation")

        with gr.Column(scale=2):
            gr.Markdown("### Emotion detection")
            mode = gr.Radio([REAL_MODE, LIVE_MODE, SIM_MODE, NO_EMOTION], value=REAL_MODE, label="Source")
            with gr.Group(visible=True) as real_group:
                sample_idx = gr.Slider(
                    0, len(get_test_data()["labels"]) - 1, value=0, step=1,
                    precision=0, label="Test utterance index", info="Real MFN runs on this utterance's features",
                )
                random_btn = gr.Button("Pick random test utterance")
                sample_preview = gr.Markdown(preview_sample(0))
            with gr.Group(visible=False) as sim_group:
                sim_label = gr.Radio(EMOTIONS, value="sad", label="Simulated label",
                                     info="Stands in for the model — labelled as simulated in the trace")
                sim_conf = gr.Slider(0.05, 1.0, value=0.8, step=0.05, label="Simulated confidence")
            with gr.Group(visible=False) as live_group:
                gr.Markdown(
                    f"**Text**: {TEXT_LABEL}. "
                    "**Audio**: emotion2vec (winning SER model, 92.7% test acc) on your mic/upload. "
                    "**Video**: no live FACET extractor exists, so video can only enter as a "
                    "clearly-labelled precomputed CMU-MOSEI segment."
                )
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                    label="Microphone / audio upload (optional)")
                live_video_chk = gr.Checkbox(value=False,
                                             label="Add video from precomputed FACET segment (not live)")
                video_idx = gr.Slider(
                    0, len(get_test_data()["labels"]) - 1, value=0, step=1, precision=0,
                    label="FACET segment index", info="CMU-MOSEI test utterance used as the video stream",
                )
            goal_box = gr.Textbox(label="Coaching goal (applied on reset)", value="reduce daily stress")

            gr.Markdown("### Pipeline inspector (last turn)")
            trace_json = gr.JSON(label="emotion → state → strategy → reply")
            memory_json = gr.JSON(label="conversation memory summary")

    inputs = [msg, chatbot, agent_state, mode, sample_idx, sim_label, sim_conf, goal_box,
              audio_in, live_video_chk, video_idx]
    outputs = [chatbot, agent_state, trace_json, memory_json, msg]
    msg.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)
    reset_btn.click(reset_conversation, [goal_box, chatbot, agent_state], outputs[:4])
    sample_idx.change(preview_sample, [sample_idx], [sample_preview])
    random_btn.click(random_index, outputs=[sample_idx])
    mode.change(toggle_mode, [mode], [real_group, sim_group, live_group])
    demo.load(lambda: (None, BehaviourChangeAgent(goal="reduce daily stress", device="cpu").memory.summary()),
              outputs=[trace_json, memory_json])


if __name__ == "__main__":
    # 0.0.0.0 so the app is reachable from outside the machine (AWS/EC2
    # security group must allow the port); loopback still works locally.
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
