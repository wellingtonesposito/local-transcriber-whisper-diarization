#!/usr/bin/env python3
"""
whisper_diarize_clean_export.py

Transcribe + diarize audio/video, lightly clean text,
OPTIONAL punctuation restoration (--punct),
OPTIONAL GPU auto-detection (--auto-gpu),
and export:

  • <prefix>.vtt
  • <prefix>_by_speaker.txt
"""

# ==============================
# Silence warnings (important)
# ==============================
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

# ==============================
# Standard Imports
# ==============================
import argparse
import json
import re
import subprocess             # <-- ADDED
from pathlib import Path      # <-- ADDED
from collections import defaultdict
from datetime import timedelta
from typing import List, Dict, Any

from tqdm import tqdm

# ==============================
# Whisper + Diarization
# ==============================
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
try:
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    _HAS_PROGRESS_HOOK = True
except Exception:
    _HAS_PROGRESS_HOOK = False

# ==============================
# Cleanup
# ==============================
from spellchecker import SpellChecker

# ==============================
# Optional punctuation
# ==============================
try:
    from whisper_punctuator import Punctuator
    _HAS_PUNCT = True
except Exception:
    _HAS_PUNCT = False

# ==============================
# GPU Auto-detection
# ==============================
import torch


def auto_select_gpu_settings():
    """Return (device, compute_type, message)."""
    if not torch.cuda.is_available():
        return ("cpu", "int8", "CUDA not available → using CPU (int8).")

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    # FP16 efficient if compute capability >= 7.0 (Volta, Turing, Ampere, Ada)
    fp16_ok = props.major >= 7

    if fp16_ok:
        return (
            "cuda",
            "float16",
            f"GPU detected: {name}\nFP16 supported → using float16."
        )
    else:
        return (
            "cuda",
            "int8_float16",
            f"GPU detected: {name}\nFP16 NOT supported → using int8_float16."
        )


# =============================================================================
# NEW: Automatic audio extraction
# =============================================================================

def ensure_wav(input_path):
    """
    Convert MP4/MKV/MOV/etc. to WAV (16k mono) if needed.
    Returns the final WAV path.
    """
    input_path = Path(input_path)

    # Already WAV → return as-is
    if input_path.suffix.lower() == ".wav":
        return str(input_path)

    wav_path = input_path.with_suffix(".wav")

    print(f"\n📥 Extracting audio from {input_path.name} → {wav_path.name} ...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path)
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise RuntimeError(f"FFmpeg failed to extract audio from {input_path}: {e}")

    print("🎵 Audio extracted successfully.\n")

    return str(wav_path)


# =============================================================================
# Utility Functions
# =============================================================================

def ts_webvtt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    td = timedelta(seconds=seconds)
    s = str(td)
    if "." not in s:
        s += ".000000"
    hms, ms = s.split(".")
    return hms.zfill(8) + "." + ms[:3]


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_adjacent(segments, max_gap=1.0):
    if not segments:
        return []
    out = [segments[0].copy()]
    for seg in segments[1:]:
        last = out[-1]
        if seg["spk"] == last["spk"] and (seg["start"] - last["end"]) < max_gap:
            last["end"] = max(last["end"], seg["end"])
            last["text"] = (last["text"] + " " + seg["text"]).strip()
        else:
            out.append(seg.copy())
    return out


# =============================================================================
# NVivo-friendly cleanup
# =============================================================================

KEEP_PAUSES = False
FILLER_PHRASES = [r"you know", r"i mean", r"sort of", r"kind of"]
FILLER_WORDS = r"(?:um+|uh+|erm)"
PAUSE_MARKERS = [
    r"\(\s*pause\s*\)",
    r"\(\s*silence\s*\)",
    r"\(\s*\d+(\.\d+)?\s*s(ec(onds)?)?\s*\)",
    r"\[\s*pause\s*\]",
    r"\[\s*silence\s*\]",
    r"\(\s*inaudible\s*\)",
    r"\(\s*overlap\s*\)",
]
_spell = SpellChecker(distance=1)

def remove_fillers(t):
    for p in FILLER_PHRASES:
        t = re.sub(rf"\b{p}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(rf"\b{FILLER_WORDS}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\blike\b", "", t, flags=re.IGNORECASE)
    return t


def remove_pauses(t):
    if KEEP_PAUSES:
        return t
    for p in PAUSE_MARKERS:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], t)
    t = re.sub(r"\b([A-Za-z])(-\1){1,}\b", r"\1", t)
    t = re.sub(r"\b([A-Za-z])\1{2,}\b", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def light_spell_correct(t):
    tokens = t.split()
    out = []
    for w in tokens:
        if not w:
            continue
        if w[0].isupper() or w.isupper() or any(c.isdigit() for c in w) or len(w) <= 3:
            out.append(w)
            continue
        try:
            cand = _spell.correction(w.lower())
        except Exception:
            cand = None
        if not cand:
            out.append(w)
            continue
        if cand != w.lower() and abs(len(cand) - len(w)) <= 2:
            out.append(cand)
        else:
            out.append(w)
    return " ".join(out)


def clean_text(t, disable_spell=False):
    t = remove_fillers(t)
    t = remove_pauses(t)
    if not disable_spell:
        t = light_spell_correct(t)
    t = re.sub(r"\s+([.,!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =============================================================================
# MAIN Pipeline
# =============================================================================

def run_pipeline(
    audio_path,
    out_prefix=None,
    model_size="small",
    device="cpu",
    compute_type="int8",
    language=None,
    beam_size=5,
    use_vad=False,
    hf_token=None,
    rename_map_path=None,
    max_merge_gap=1.0,
    disable_spell=False,
    punct=False,
):

    # Create output folder for VTT files
    output_folder = "Local - Transcripts to revise"
    os.makedirs(output_folder, exist_ok=True)

    base = out_prefix or os.path.splitext(os.path.basename(audio_path))[0]
    vtt_path = os.path.join(output_folder, base + ".vtt")

    # TXT output disabled
    txt_path = None

    print(f"\n🔊 Transcribing: {audio_path}")
    print(f"   Model: {model_size}")
    print(f"   Device: {device}")
    print(f"   Compute Type: {compute_type}\n")

    # --------------------
    # 1) Load Whisper
    # --------------------
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    seg_gen, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=use_vad,
        beam_size=beam_size,
    )

    duration_sec = getattr(info, "duration", 0.0) or 0.0
    total_min = max(1, int(duration_sec // 60))

    # --------------------
    # Progress bar
    # --------------------
    pbar = tqdm(total=total_min, unit="min", desc="🎧 Transcribing (audio minutes)")
    pbar.set_postfix_str("min transcribed/total [time spent/eta, rate]")

    asr = []
    last_end = 0.0

    for s in seg_gen:
        asr.append({"start": s.start, "end": s.end, "text": (s.text or "").strip()})
        progressed = max(0.0, s.end - last_end)
        if progressed > 0:
            last_end = s.end
            m = int(last_end // 60)
            if m > pbar.n:
                pbar.n = min(m, total_min)
                pbar.refresh()

    pbar.n = total_min
    pbar.refresh()
    pbar.close()

    print(f"\n✅ Transcription complete ({len(asr)} segments)\n")

    # --------------------
    # 2) Diarization
    # --------------------
    print("👥 Running diarization...")
    try:
        if hf_token:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                use_auth_token=hf_token)
        else:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        if _HAS_PROGRESS_HOOK:
            with ProgressHook() as hook:
                diar = pipeline(audio_path, hook=hook)
        else:
            with tqdm(total=100, desc="👥 Diarization", unit="%") as dbar:
                diar = pipeline(audio_path)
                dbar.update(100)
    except Exception as e:
        raise RuntimeError(f"Diarization failed: {e}")

    print("✅ Diarization complete.\n")

    # --------------------
    # 3) Build speaker turns
    # --------------------
    turns = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "spk": str(spk)
        })

    # --------------------
    # 4) Assign segments to speaker
    # --------------------
    labeled = []
    for s in asr:
        if not s["text"]:
            continue
        best = max(turns, key=lambda t: overlap(
            s["start"], s["end"], t["start"], t["end"]
        ))
        labeled.append({
            "start": s["start"],
            "end": s["end"],
            "spk": best["spk"],
            "text": s["text"],
        })

    # --------------------
    # 5) Merge + clean
    # --------------------
    merged = merge_adjacent(labeled, max_merge_gap)

    for seg in merged:
        seg["text"] = clean_text(seg["text"], disable_spell=disable_spell)

    # --------------------
    # 6) Optional punctuation
    # --------------------
    if punct:
        if not _HAS_PUNCT:
            raise RuntimeError("whisper-punctuator not installed. Run: pip install whisper-punctuator")
        print("🔤 Adding punctuation...")
        punct_model = Punctuator(language="en", punctuations=",.?!")

        for seg in merged:
            try:
                seg["text"] = punct_model.punctuate(audio_path, seg["text"])
            except Exception:
                pass

    # --------------------
    # 7) Optional speaker renaming
    # --------------------
    rename_map = {}
    if rename_map_path and os.path.exists(rename_map_path):
        with open(rename_map_path, "r", encoding="utf-8") as f:
            rename_map = json.load(f)

    def pretty_spk(s):
        return rename_map.get(s, s)

    # --------------------
    # 8) Write VTT
    # --------------------
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in merged:
            f.write(f"{ts_webvtt(s['start'])} --> {ts_webvtt(s['end'])}\n")
            f.write(f"{pretty_spk(s['spk'])}: {s['text']}\n\n")

    # --------------------
    # 9) Write grouped TXT
    # --------------------
    # TXT generation disabled
    order = list({pretty_spk(s["spk"]) for s in merged})


    print("🎉 Done!")
    print("Saved:")
    print(" •", vtt_path)
    # print(" •", txt_path)
    print("Speakers:", ", ".join(order))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("audio", help="Audio/video file path")
    a.add_argument("--out-prefix")
    a.add_argument("--model-size", default="small")
    a.add_argument("--device", default="cpu")          # <-- fixed
    a.add_argument("--compute-type", default="int8")
    a.add_argument("--language", default=None)
    a.add_argument("--beam-size", type=int, default=5)
    a.add_argument("--vad", action="store_true")
    a.add_argument("--hf-token", default=None)
    a.add_argument("--rename-map", default=None)
    a.add_argument("--max-merge-gap", type=float, default=1.0)
    a.add_argument("--no-spell", action="store_true")
    a.add_argument("--punct", action="store_true",
                   help="Add punctuation using whisper-punctuator")
    a.add_argument("--auto-gpu", action="store_true",
                   help="Automatically select best GPU settings.")
    return a.parse_args()


def main():
    args = parse_args()

    # Auto GPU selection
    if args.auto_gpu:
        device, ctype, msg = auto_select_gpu_settings()
        print("\n⚡ Auto GPU Selection:")
        print(msg)
        print(f"Selected device={device}, compute_type={ctype}\n")

        args.device = device
        args.compute_type = ctype

    # NEW: Automatic audio extraction
    audio_input = ensure_wav(args.audio)

    run_pipeline(
        audio_path=audio_input,
        out_prefix=args.out_prefix,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        use_vad=args.vad,
        hf_token=args.hf_token,
        rename_map_path=args.rename_map,
        max_merge_gap=args.max_merge_gap,
        disable_spell=args.no_spell,
        punct=args.punct,
    )

if __name__ == "__main__":
    main()
