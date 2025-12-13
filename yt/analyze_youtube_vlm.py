import os
import io
import json
import cv2
import sqlite3
import tempfile
import subprocess
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import yt_dlp
import requests

# Optional LLM backends
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    HAS_QWEN = True
except Exception:
    HAS_QWEN = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


# -----------------------------
# Config
# -----------------------------
OUTPUT_DIR = Path("vlm_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = OUTPUT_DIR / "video_analytics.db"

# Sampling parameters
SCENE_CHANGE_METHOD = "content-diff"  # "content-diff" (opencv histogram) or "hist-threshold" (cv2)
HIST_THRESHOLD = 0.6  # for hist-threshold
CONTENT_DIFF_THRESHOLD = 0.05  # for content-diff
FALLBACK_EVERY_N_SECONDS = 15  # if no scene cut found for this long, force a frame

# VLM backends
# Set one of: "qwen2vl" or "openai"
VLM_BACKEND = os.environ.get("VLM_BACKEND", "qwen2vl")
# For Qwen2-VL model path (local or HF)
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
# For OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------
# Utilities
# -----------------------------

def ensure_video_downloaded(url: str, out_dir: Path) -> Path:
    """
    Download best video+audio to out_dir with yt-dlp.
    Returns path to the downloaded file.
    """
    ydl_opts = {
        "outtmpl": str(out_dir / "%(title).80s.%(id)s.%(ext)s"),
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": False,
        "nocheckcertificate": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return Path(filename)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    scene_change_method: str = SCENE_CHANGE_METHOD,
    hist_threshold: float = HIST_THRESHOLD,
    content_diff_threshold: float = CONTENT_DIFF_THRESHOLD,
    fallback_every_n_seconds: int = FALLBACK_EVERY_N_SECONDS
) -> List[Dict[str, Any]]:
    """
    Extract key frames based on scene changes + fallback sampling.
    Returns a list of dicts:
      {
        "scene_id": int,
        "frame_idx": int,
        "timestamp_s": float,
        "image_path": str,
        "change_reason": str  # "scene-cut" or "fallback"
      }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    last_frame = None
    last_hist = None
    last_keyframe_info = None
    frames = []
    scene_id = 0
    last_keyframe_time = 0.0

    out_frames_dir = out_dir / "frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(frame_idx: int, ts: float, change_reason: str, frame):
        nonlocal scene_id, last_keyframe_time
        if ts - last_keyframe_time >= fallback_every_n_seconds:
            change_reason = "fallback"
        filename = out_frames_dir / f"scene_{scene_id:04d}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(filename), frame)
        frames.append({
            "scene_id": scene_id,
            "frame_idx": frame_idx,
            "timestamp_s": ts,
            "image_path": str(filename),
            "change_reason": change_reason
        })
        last_keyframe_time = ts
        return filename

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps

        if scene_change_method == "hist-threshold":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [50,60], [0,180,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            if last_hist is not None:
                diff = 1 - cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                if diff >= hist_threshold:
                    scene_id += 1
                    save_frame(frame_idx, ts, change_reason="scene-cut", frame=frame)
            last_hist = hist

        elif scene_change_method == "content-diff":
            if last_frame is not None:
                diff = cv2.absdiff(last_frame, frame)
                non_zero = cv2.countNonZero(diff)
                total_pixels = diff.shape[0] * diff.shape[1]
                ratio = (non_zero / float(total_pixels)) if total_pixels > 0 else 0.0
                if ratio >= content_diff_threshold:
                    scene_id += 1
                    save_frame(frame_idx, ts, change_reason="scene-cut", frame=frame)
            last_frame = frame

        else:
            raise ValueError("Unknown scene change method")

        frame_idx += 1

        # fallback periodic frame if no cut for long time
        if (ts - last_keyframe_time) >= fallback_every_n_seconds:
            scene_id += 1
            save_frame(frame_idx-1, ts, change_reason="fallback", frame=frame)

    cap.release()
    # Ensure we always include the first frame
    if frames and frames[0]["frame_idx"] != 0:
        cap2 = cv2.VideoCapture(str(video_path))
        ret, first = cap2.read()
        if ret:
            save_frame(0, 0.0, change_reason="fallback", frame=first)
        cap2.release()

    # sort frames by timestamp
    frames.sort(key=lambda x: x["timestamp_s"])
    # reset scene_id as strictly increasing w.r.t sorted frames
    for i, f in enumerate(frames):
        f["scene_id"] = i

    return frames


def image_to_data_url(image_path: Path) -> str:
    """
    For OpenAI (or other HTTP) you may need a data URL.
    """
    import base64
    ext = image_path.suffix.lower().replace(".", "")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


# -----------------------------
# VLM Backends
# -----------------------------

class Qwen2VLBackend:
    def __init__(self, model_id: str):
        if not HAS_QWEN:
            raise RuntimeError("transformers/accelerate not installed")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    def chat(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024) -> str:
        """
        messages: [{'role': 'user', 'content': [{'type':'text', 'text':...}, {'type':'image_url','image_url':{'url': ...}}]}]
        Qwen2VL expects image as <image> token if sending raw data, but we can also pass
        a PIL image via processor. Simpler: pass a direct path and rely on processor to load.
        """
        # Extract image path if present
        image_path = None
        for m in messages:
            if m.get("role") == "user":
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "image_url":
                        url = c.get("image_url", {}).get("url", "")
                        if url.startswith("file://"):
                            image_path = url.replace("file://", "")
                        elif url.startswith("data:"):
                            # decode and write temp file (optional)
                            import base64, tempfile
                            header, b64 = url.split(",", 1)
                            img_bytes = base64.b64decode(b64)
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            tmp.write(img_bytes)
                            tmp.flush()
                            image_path = tmp.name
        if image_path is None:
            raise ValueError("No image path provided in messages")

        # Build prompt text
        text_prompt = ""
        for m in messages:
            if m.get("role") == "user":
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        text_prompt += c.get("text", "")
            elif m.get("role") == "system":
                text_prompt += m.get("content", "") + "\n"

        inputs = self.processor(text=[text_prompt], images=[image_path], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out


class OpenAIBackend:
    def __init__(self, api_key: str, model: str):
        if not HAS_OPENAI:
            raise RuntimeError("openai not installed")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024) -> str:
        # OpenAI expects content with text and image_url
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content


def get_vlm_backend() -> Any:
    if VLM_BACKEND == "qwen2vl":
        return Qwen2VLBackend(QWEN_MODEL_ID)
    elif VLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIBackend(OPENAI_API_KEY, OPENAI_MODEL)
    else:
        raise ValueError("Unknown VLM_BACKEND")


# -----------------------------
# Prompts (three "agents")
# -----------------------------

SYSTEM_COMMON = (
    "You are a careful, exhaustive visual analyst. "
    "Never make up details. If uncertain, say 'uncertain'. "
    "Use units, counts, and exact wording when possible."
)

SCANNER_PROMPT = (
    SYSTEM_COMMON +
    "You are the SCANNER agent. Produce a structured report of this single video frame.\n"
    "Describe in rich detail: actors/people, objects, text, UI, background, environment, lighting, composition, camera angle, any small changes.\n"
    "Return strict JSON with keys: "
    "objects (list of {name, attributes, bbox [x1,y1,x2,y2] if confident, occluded bool}), "
    "text_spotted (list of {content, bbox if confident}), "
    "ui_elements (list of {type, description}), "
    "scene_summary (string <80 words), "
    "notable_details (string), "
    "quality (good|medium|poor), "
    "objects_count (dict name->count). "
    "bbox coordinates are absolute pixel values in the original image size."
)

TRACKER_PROMPT = (
    SYSTEM_COMMON +
    "You are the TRACKER agent. Compare this frame to the previous frame and report only CHANGES.\n"
    "Produce strict JSON with keys: "
    "intro_objects (list of {name, attributes, bbox if confident}), "
    "removed_objects (list of {name, attributes, bbox if confident}), "
    "moved_objects (list of {name, from_bbox [x1,y1,x2,y2], to_bbox [x1,y1,x2,y2], delta_pixels [dx,dy]}), "
    "changed_attributes (list of {name, attributes_before, attributes_after, bbox if confident}), "
    "text_changes (list of {change_type: added|removed|changed, content_before, content_after, bbox if confident}), "
    "global_events (list of {event_type, description}), "
    "notes (string). "
    "When uncertain, set change_type as 'uncertain' and explain why."
)

QA_PROMPT = (
    SYSTEM_COMMON +
    "You are the Q&A agent. Generate 2-3 short, answerable questions that would reduce ambiguity "
    "if answered by a viewer or downstream system. Prefer questions about: object identity, counts, "
    "text content, UI state, camera perspective, lighting, or actions implied.\n"
    "Return strict JSON: { 'questions': [ { 'q': '...', 'type': 'count|identity|text|state|other', 'hint': '...' } ] }"
)


# -----------------------------
# DB schema
# -----------------------------
def init_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scenes (
        scene_id INTEGER PRIMARY KEY,
        start_time_s REAL,
        end_time_s REAL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS frames (
        frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id INTEGER,
        frame_idx INTEGER,
        timestamp_s REAL,
        image_path TEXT,
        scanner_json TEXT,
        tracker_json TEXT,
        qa_json TEXT,
        FOREIGN KEY(scene_id) REFERENCES scenes(scene_id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id INTEGER,
        name TEXT,
        attributes TEXT,
        bbox TEXT,
        introduced_at INTEGER,
        removed_at INTEGER,
        FOREIGN KEY(scene_id) REFERENCES scenes(scene_id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id INTEGER,
        frame_id INTEGER,
        type TEXT,
        description TEXT,
        payload TEXT,
        FOREIGN KEY(scene_id) REFERENCES scenes(scene_id),
        FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qa_questions (
        qa_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id INTEGER,
        frame_id INTEGER,
        question TEXT,
        qtype TEXT,
        hint TEXT,
        FOREIGN KEY(scene_id) REFERENCES scenes(scene_id),
        FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
    )
    """)
    conn.commit()
    conn.close()


def db_insert_frame(conn, scene_id: int, frame_idx: int, ts: float, img_path: str,
                    scanner_json: Dict[str, Any],
                    tracker_json: Dict[str, Any],
                    qa_json: Dict[str, Any]) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO frames (scene_id, frame_idx, timestamp_s, image_path, scanner_json, tracker_json, qa_json) VALUES (?,?,?,?,?,?,?)",
        (scene_id, frame_idx, ts, img_path, json.dumps(scanner_json), json.dumps(tracker_json), json.dumps(qa_json))
    )
    conn.commit()
    return cur.lastrowid


def db_insert_entities(conn, scene_id: int, introduced: List[Dict[str, Any]], removed: List[Dict[str, Any]], moved: List[Dict[str, Any]]):
    cur = conn.cursor()
    for e in introduced:
        cur.execute(
            "INSERT INTO entities (scene_id, name, attributes, bbox, introduced_at, removed_at) VALUES (?,?,?,?,?,?)",
            (scene_id, e.get("name", ""), json.dumps(e.get("attributes", {})), json.dumps(e.get("bbox", [])), scene_id, None)
        )
    for e in removed:
        cur.execute(
            "INSERT INTO entities (scene_id, name, attributes, bbox, introduced_at, removed_at) VALUES (?,?,?,?,?,?)",
            (scene_id, e.get("name", ""), json.dumps(e.get("attributes", {})), json.dumps(e.get("bbox", [])), scene_id, scene_id)
        )
    # moved: if a real tracker existed, we'd update; here we just log an event
    for m in moved:
        cur.execute(
            "INSERT INTO events (scene_id, frame_id, type, description, payload) VALUES (?,?,?,?,?)",
            (scene_id, None, "moved", f"{m.get('name','')} moved", json.dumps(m))
        )
    conn.commit()


def db_insert_qa(conn, scene_id: int, frame_id: int, qa_json: Dict[str, Any]):
    cur = conn.cursor()
    for q in qa_json.get("questions", []):
        cur.execute(
            "INSERT INTO qa_questions (scene_id, frame_id, question, qtype, hint) VALUES (?,?,?,?,?)",
            (scene_id, frame_id, q.get("q",""), q.get("type",""), q.get("hint",""))
        )
    conn.commit()


def db_insert_global_events(conn, scene_id: int, frame_id: Optional[int], events: List[Dict[str, Any]]):
    cur = conn.cursor()
    for ev in events:
        cur.execute(
            "INSERT INTO events (scene_id, frame_id, type, description, payload) VALUES (?,?,?,?,?)",
            (scene_id, frame_id, ev.get("event_type",""), ev.get("description",""), json.dumps(ev))
        )
    conn.commit()


# -----------------------------
# Core pipeline
# -----------------------------

@dataclass
class PreviousFrameState:
    prev_scanner: Optional[Dict[str, Any]] = None
    prev_path: Optional[str] = None
    prev_timestamp: Optional[float] = None


def build_messages(image_path: Path, system_prompt: str, user_text: str) -> List[Dict[str, Any]]:
    # For Qwen: use "image_url" with file:// path to keep it simple
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
        ]}
    ]


def run_agent(vlm, image_path: Path, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    messages = build_messages(image_path, system_prompt, user_prompt)
    raw = vlm.chat(messages, max_new_tokens=1200)
    try:
        j = json.loads(raw)
    except Exception:
        # try to recover JSON if extra text was generated
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            j = json.loads(raw[start:end])
        else:
            raise ValueError("Agent did not return valid JSON")
    return j


def analyze_youtube(url: str):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path = ensure_video_downloaded(url, tmp)
        frames = extract_frames(video_path, OUTPUT_DIR)

        conn = sqlite3.connect(str(DB_PATH))
        init_db(DB_PATH)

        vlm = get_vlm_backend()
        prev_state = PreviousFrameState()
        all_narratives = []

        for i, f in enumerate(frames):
            scene_id = f["scene_id"]
            frame_idx = f["frame_idx"]
            ts = f["timestamp_s"]
            img_path = Path(f["image_path"])

            # 1) Scanner
            scanner = run_agent(vlm, img_path, SYSTEM_COMMON, SCANNER_PROMPT)

            # 2) Tracker
            if prev_state.prev_scanner is not None:
                tracker = run_agent(vlm, img_path, SYSTEM_COMMON, TRACKER_PROMPT)
            else:
                tracker = {
                    "intro_objects": scanner.get("objects", []),
                    "removed_objects": [],
                    "moved_objects": [],
                    "changed_attributes": [],
                    "text_changes": [],
                    "global_events": [],
                    "notes": "First frame in video."
                }

            # 3) Q&A
            qa = run_agent(vlm, img_path, SYSTEM_COMMON, QA_PROMPT)

            # Persist to DB
            frame_id = db_insert_frame(conn, scene_id, frame_idx, ts, str(img_path), scanner, tracker, qa)
            db_insert_entities(conn, scene_id, tracker.get("intro_objects", []), tracker.get("removed_objects", []), tracker.get("moved_objects", []))
            db_insert_qa(conn, scene_id, frame_id, qa)
            db_insert_global_events(conn, scene_id, frame_id, tracker.get("global_events", []))

            # Build a narrative for the scene (merge scanner + tracker)
            narrative = {
                "scene_id": scene_id,
                "timestamp_s": ts,
                "change_reason": f.get("change_reason", ""),
                "scene_summary": scanner.get("scene_summary", ""),
                "objects_count": scanner.get("objects_count", {}),
                "tracker_notes": tracker.get("notes", ""),
                "events": tracker.get("global_events", [])
            }
            all_narratives.append(narrative)

            # update prev state
            prev_state.prev_scanner = scanner
            prev_state.prev_path = str(img_path)
            prev_state.prev_timestamp = ts

            print(f"Scene {scene_id} | t={ts:.2f}s | reason={f.get('change_reason','n/a')} | "
                  f"objects={len(scanner.get('objects', []))} | events={len(tracker.get('global_events', []))}")

        conn.close()

        # Save merged narratives
        with open(OUTPUT_DIR / "scene_narratives.jsonl", "w", encoding="utf-8") as fh:
            for n in all_narratives:
                fh.write(json.dumps(n, ensure_ascii=False) + "\n")

        print("\nDone. Outputs:")
        print("- SQLite DB:", DB_PATH)
        print("- Frames dir:", OUTPUT_DIR / "frames")
        print("- Scene narratives:", OUTPUT_DIR / "scene_narratives.jsonl")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_youtube_vlm.py '<YouTube URL>'")
        sys.exit(1)
    analyze_youtube(sys.argv[1])