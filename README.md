# Whisper Diarize Clean Export

This script transcribes an audio or video file with `faster-whisper`, assigns speaker labels with `pyannote.audio`, lightly cleans the transcript text, and writes a WebVTT transcript file.

## What the script does

Given an input file such as `.wav`, `.mp4`, `.mov`, or `.mkv`, the script:

1. Converts non-WAV input into a 16 kHz mono WAV with `ffmpeg`.
2. Runs Whisper transcription with `faster-whisper`.
3. Runs speaker diarization with `pyannote.audio`.
4. Assigns each transcript segment to the speaker turn with the greatest overlap.
5. Merges nearby same-speaker segments.
6. Removes filler words and pause markers, and optionally runs light spell correction.
7. Optionally restores punctuation.
8. Writes a `.vtt` transcript into `Local - Transcripts to revise`.

## Important behavior notes

- The script currently writes only a `.vtt` file.
- The top-level docstring still mentions a speaker-grouped `.txt` export, but that export is disabled in the current code.
- If you pass a video file, the extracted `.wav` is created next to the original input file.
- Speaker names default to labels such as `SPEAKER_00`, unless you supply a rename map JSON file.

## Prerequisites

You need:

- Python 3.12
- `ffmpeg` available on your system `PATH`
- A Hugging Face token if diarization access requires authentication
- Accepted access terms for the `pyannote/speaker-diarization-3.1` model on Hugging Face

Run commands from the project root, using PowerShell on Windows or Terminal on macOS.

## Setup

### Windows

#### 1. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

#### 2. Install Python packages

```powershell
pip install -r requirements.txt
```

If you want GPU support, install a matching PyTorch build first or replace the existing `torch` install with the correct CUDA/CPU build for your machine.

Optional package for `--punct`:

```powershell
pip install whisper-punctuator
```

#### 3. Install `ffmpeg`

Make sure `ffmpeg` runs from the terminal:

```powershell
ffmpeg -version
```

If this fails, install `ffmpeg` and add it to your `PATH` before using the script.

#### 4. Set up Hugging Face access for diarization

If diarization fails because the model requires authentication:

1. Create a Hugging Face access token.
2. Accept the model terms for `pyannote/speaker-diarization-3.1`.
3. Either log in once:

```powershell
huggingface-cli login
```

4. Or pass the token at runtime with `--hf-token`.

### macOS

#### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Python packages

```bash
pip install -r requirements.txt
```

Optional package for `--punct`:

```bash
pip install whisper-punctuator
```

If you want hardware acceleration on macOS, verify what your installed PyTorch build supports. The current `--auto-gpu` logic only checks CUDA, so macOS users should normally run with `--device cpu` unless they adapt the script for MPS.

#### 3. Install `ffmpeg`

If you use Homebrew:

```bash
brew install ffmpeg
ffmpeg -version
```

#### 4. Set up Hugging Face access for diarization

```bash
huggingface-cli login
```

Or pass a token at runtime with `--hf-token`.

## Basic usage

From the project root:

```powershell
venv\Scripts\activate
python whisper_diarize_clean_export.py "input-file.mp4"
```

A more typical GPU example:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --device cuda --compute-type float16 --vad
```

Example with punctuation:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --device cuda --compute-type float16 --vad --punct
```

Example using automatic GPU selection:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --auto-gpu --vad
```

macOS example:

```bash
source venv/bin/activate
python3 whisper_diarize_clean_export.py "input-file.mp4" --device cpu --compute-type int8 --vad
```

## Output files

The script creates:

- `Local - Transcripts to revise/<prefix>.vtt`

If the input was not already a WAV, it also creates:

- `<input_basename>.wav` beside the original source file

The output filename prefix defaults to the input filename stem, unless overridden with `--out-prefix`.

## Command-line flags

Full CLI shape:

```text
python whisper_diarize_clean_export.py [options] audio
```

### Positional argument

#### `audio`

Path to the input audio or video file.

- Required.
- Can be a WAV directly.
- Can also be a video/container file such as `.mp4`, `.mov`, or `.mkv`; those will be converted to WAV first.

### Optional flags

#### `--out-prefix OUT_PREFIX`

Overrides the output filename stem.

Example:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --out-prefix review_copy
```

This writes:

```text
Local - Transcripts to revise/review_copy.vtt
```

Use this when:

- You want a clearer transcript filename than the source media filename.
- You are generating multiple transcript variants from the same recording.

#### `--model-size MODEL_SIZE`

Chooses the Whisper model size passed to `faster-whisper`.

Default:

```text
small
```

Typical values include `tiny`, `base`, `small`, `medium`, and `large`, depending on what your `faster-whisper` install supports.

Tradeoff:

- Smaller models are faster and lighter.
- Larger models usually improve transcription quality, especially on harder audio, at the cost of more memory and slower runtime.

#### `--device DEVICE`

Controls where Whisper runs.

Examples:

- `cpu`
- `cuda`

Default:

```text
cpu
```

Use `cuda` only when CUDA-enabled PyTorch and a compatible NVIDIA GPU are available.

#### `--compute-type COMPUTE_TYPE`

Controls the numeric precision used by `faster-whisper`.

Default:

```text
int8
```

Common values:

- `int8`: lower memory use, good CPU default
- `float16`: common GPU choice on newer NVIDIA GPUs
- `int8_float16`: mixed option that can help on some GPUs

Practical guidance:

- CPU: `--device cpu --compute-type int8`
- Modern NVIDIA GPU: `--device cuda --compute-type float16`
- Older or more limited GPU: `--device cuda --compute-type int8_float16`

#### `--language LANGUAGE`

Forces a language code for transcription instead of letting Whisper detect it.

Examples:

- `en`
- `es`
- `pt`

Use this when:

- The audio is clearly one language and you want more stable results.
- Auto-detection gets the language wrong.

#### `--beam-size BEAM_SIZE`

Sets Whisper decoding beam size.

Default:

```text
5
```

Tradeoff:

- Higher values may improve transcript quality slightly.
- Higher values also increase runtime.

If you want faster runs, try a smaller beam size such as `1` or `3`.

#### `--vad`

Enables Whisper voice activity detection filtering during transcription.

This can help:

- Reduce low-value segments from silence or noise
- Improve transcript segmentation on recordings with pauses

This may be useful for interviews or screen recordings with long quiet stretches.

#### `--hf-token HF_TOKEN`

Passes a Hugging Face token directly into the diarization model loader.

Use this when:

- You have not logged in with `huggingface-cli login`
- You want to keep authentication explicit per run

Example:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --hf-token YOUR_TOKEN_HERE
```

Security note:

- Do not commit tokens to source control.
- Avoid storing raw tokens in shared notes or command history if this folder is shared.

#### `--rename-map RENAME_MAP`

Path to a JSON file that maps diarization speaker labels to friendlier names.

Example JSON:

```json
{
  "SPEAKER_00": "Participant",
  "SPEAKER_01": "Researcher"
}
```

Example command:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --rename-map speaker_names.json
```

This affects the speaker label written into the `.vtt` output.

#### `--max-merge-gap MAX_MERGE_GAP`

Controls how aggressively adjacent same-speaker transcript segments are merged.

Default:

```text
1.0
```

Meaning:

- If two neighboring segments belong to the same speaker and the silence/gap between them is less than this many seconds, the script merges them.

Lower values:

- Keep more, shorter subtitle blocks

Higher values:

- Produce fewer, longer blocks

This can materially change subtitle readability.

#### `--no-spell`

Disables the script's light spell correction step.

By default, the script tries to lightly correct some lowercase words after filler/pause cleanup. It already avoids changing:

- Capitalized words
- All-uppercase words
- Words containing digits
- Very short words

Use `--no-spell` when:

- Proper nouns are getting changed incorrectly
- Domain-specific vocabulary is important
- You want the rawer transcript text for manual review

#### `--punct`

Enables punctuation restoration with `whisper-punctuator`.

Behavior:

- This runs after transcription, diarization, merging, and cleanup.
- If `whisper-punctuator` is not installed, the script raises an error.

Use this when:

- Your transcript is mostly unpunctuated and hard to read
- You want easier manual review in subtitle form

Caution:

- Punctuation restoration is heuristic, so it can occasionally add awkward punctuation.

#### `--auto-gpu`

Automatically selects `--device` and `--compute-type` based on whether CUDA is available and whether the first GPU appears suitable for FP16.

Behavior in the current script:

- No CUDA available: selects `device=cpu`, `compute_type=int8`
- CUDA with compute capability 7.0 or newer: selects `device=cuda`, `compute_type=float16`
- Older CUDA GPU: selects `device=cuda`, `compute_type=int8_float16`

This flag overrides manual `--device` and `--compute-type` choices after parsing arguments.

Use it when:

- You want a safer default without checking GPU details manually
- The script may run on different machines

## Recommended command patterns

### CPU-only

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --device cpu --compute-type int8 --vad
```

### GPU with explicit settings

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --device cuda --compute-type float16 --vad
```

### GPU with automatic settings

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --auto-gpu --vad
```

### Rename speakers and keep a custom output name

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --auto-gpu --vad --rename-map speaker_names.json --out-prefix cleaned_copy
```

## Suggested workflow

1. Put the media file anywhere you want to process it from.
2. Activate the virtual environment.
3. Run the script on one recording.
4. Review the `.vtt` file created in `Local - Transcripts to revise`.
5. If speaker labels are too generic, create a rename-map JSON file and rerun.
6. If text cleanup feels too aggressive, rerun with `--no-spell`.
7. If transcript blocks are too fragmented or too long, tune `--max-merge-gap`.

## Troubleshooting

### `ffmpeg` not found

Install `ffmpeg` and make sure it is on `PATH`.

### Diarization fails with authentication or access errors

- Accept the model terms on Hugging Face.
- Log in with `huggingface-cli login`.
- Or pass `--hf-token`.

### `--punct` fails

Install the optional package:

```powershell
pip install whisper-punctuator
```

### GPU run fails

Check:

- CUDA-compatible NVIDIA drivers are installed
- Your PyTorch build matches your CUDA setup
- `torch.cuda.is_available()` is true in the same virtual environment

### Output looks over-cleaned

Try:

- `--no-spell`
- a smaller `--max-merge-gap`
- manual review of the `.vtt` in `Local - Transcripts to revise`

## Example rename map file

Save as `speaker_names.json`:

```json
{
  "SPEAKER_00": "Participant",
  "SPEAKER_01": "Interviewer"
}
```

Then run:

```powershell
python whisper_diarize_clean_export.py "input-file.mp4" --auto-gpu --vad --rename-map speaker_names.json
```
