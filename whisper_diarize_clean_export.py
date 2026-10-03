#!/usr/bin/env python3
"""
Backwards-compatible entry point for the original script.

The implementation now lives in the `transcriber` package (see README). This wrapper keeps the old
command line working and writes `Local - Transcripts to revise/<prefix>.vtt` as before.

Differences from the old script (all deliberate, see README "What changed"):
  * --device / --compute-type / --auto-gpu are ignored: the engine and hardware are chosen automatically
    (Apple Silicon GPU via mlx-whisper, NVIDIA CUDA or CPU via faster-whisper).
  * --no-spell and --punct are ignored: spell-correction is not applied (it corrupted names and non-English
    text) and Whisper already punctuates.
  * Fillers ("um", "uh") and stutters are still removed from the exported text, like before, but "like",
    "you know", "I mean" etc. are no longer deleted.
  * The extracted .wav is no longer written next to your source file.

Prefer `transcriber serve` for the web app or `transcriber run` for the new CLI.
"""
import argparse
import os
import sys


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("audio", help="Audio/video file path")
    a.add_argument("--out-prefix")
    a.add_argument("--model-size", default="small")
    a.add_argument("--device", default=None, help="ignored (automatic)")
    a.add_argument("--compute-type", default=None, help="ignored (automatic)")
    a.add_argument("--language", default=None)
    a.add_argument("--beam-size", type=int, default=5)
    a.add_argument("--vad", action="store_true")
    a.add_argument("--hf-token", default=None)
    a.add_argument("--rename-map", default=None)
    a.add_argument("--max-merge-gap", type=float, default=1.0)
    a.add_argument("--no-spell", action="store_true", help="ignored (spell-correction removed)")
    a.add_argument("--punct", action="store_true", help="ignored (Whisper already punctuates)")
    a.add_argument("--auto-gpu", action="store_true", help="ignored (automatic)")
    args = a.parse_args()

    ignored = [f for f, v in (("--device", args.device), ("--compute-type", args.compute_type),
                              ("--no-spell", args.no_spell), ("--punct", args.punct), ("--auto-gpu", args.auto_gpu)) if v]
    if ignored:
        print(f"Note: {', '.join(ignored)} no longer have any effect (see the header of this file).", file=sys.stderr)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    from transcriber.cli import main as cli_main

    argv = ["run", args.audio, "--formats", "vtt", "--clean", "--model-size", args.model_size,
            "--beam-size", str(args.beam_size), "--max-merge-gap", str(args.max_merge_gap)]
    if args.out_prefix:
        argv += ["--out-prefix", args.out_prefix]
    if args.language:
        argv += ["--language", args.language]
    if args.vad:
        argv += ["--vad"]
    if args.rename_map:
        argv += ["--rename-map", args.rename_map]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
