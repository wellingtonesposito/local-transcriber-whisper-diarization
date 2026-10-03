# Transcriber

A local web app for turning interview recordings into speaker-labelled transcripts, built on
[Whisper](https://github.com/SYSTRAN/faster-whisper) (speech to text) and
[pyannote](https://github.com/pyannote/pyannote-audio) (who spoke when). Everything runs on your computer: recordings and
transcripts are never uploaded anywhere. Exports are designed for qualitative analysis software (NVivo, ATLAS.ti, MAXQDA,
Dedoose) as well as plain text, Word, PDF and captions.

1. **Create a project** for a study.
2. **Drag and drop** one or many audio/video files.
3. **Transcribe**, with live progress.
4. **Review**: play the recording next to the text, correct words, name the speakers.
5. **Export** to the formats your analysis tool needs.

## Quick start

Requirements: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/) (or plain `pip`). `ffmpeg` is bundled; a system
ffmpeg is used if you already have one.

```bash
# Apple Silicon Mac (adds the fast GPU engine)
uv venv --python 3.12 && uv pip install -e ".[asr,mlx]"

# Windows / Linux / Intel Mac
uv venv --python 3.12 && uv pip install -e ".[asr]"
```

On an NVIDIA machine, install the matching CUDA build of PyTorch first (see pytorch.org).

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
transcriber doctor                 # checks the installation
transcriber serve                  # opens http://127.0.0.1:8765
```

### One-time setup for speaker identification

Telling speakers apart uses pyannote's models. They are free but gated by Hugging Face:

1. Create a free account and a **read** token at <https://huggingface.co/settings/tokens>.
2. Accept the terms on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
3. Open **Setup** in the app, paste the token, click **Test access**.

The token is stored in your data folder (readable only by you) and only ever sent to Hugging Face. After the first
download the models are cached and speaker identification works offline. Without a token you can still transcribe by
turning off **Identify speakers** (everything is then labelled as one speaker).

## Using the app

**Project page.** Drop files on the drop zone (or click it). New projects start on the **Interview** preset (2 speakers) with the `turbo` model; pick another preset (Focus group, Quick draft) or adjust the settings, then press **Transcribe**. Several files
are queued and processed one after another; models stay loaded between files, so a batch is faster than running files
separately. You can cancel a file at any time; if a run fails (for example because the Hugging Face terms were not yet
accepted) the finished stages are kept and **Retry** resumes where it stopped.

| Setting | What it does |
|---|---|
| Accuracy (model) | `turbo` by default: fast and accurate. `small` is faster; `large-v3` is the most accurate. |
| Language | Auto-detect, or force one for more stable results. |
| Speakers (exact number) | If you know it (most interviews: 2), say so. It improves speaker accuracy. Min/max are under *Advanced*. |
| Keep "um", "uh" | Nudges Whisper to transcribe disfluencies instead of silently dropping them. |
| Skip silence (VAD) | Fewer made-up words in quiet stretches. |
| Vocabulary / context | Names, jargon, acronyms to help spelling. |

**Transcripts screen.** Open it from the project in the sidebar (or the **Transcripts** button). It lists every transcribed
recording with its state: **To review**, **In review** (you have made corrections or named speakers) or **Reviewed** (you
marked it done). Filter by state, open a transcript, mark it reviewed, or export one or all of them. Recordings stay on the
**Recordings** screen, which is only about uploading and transcribing.

**Review page.** The speakers (rename them here; names apply everywhere, including exports) and their share of the talking
time are in a strip at the top. Each speaker's consecutive lines are shown as one block. While the recording plays, the
word being spoken is highlighted and the view follows along (scroll yourself and it steps back for a few seconds). Click any
word to play from it, or click a timestamp to jump to the start of a block. Double-click a word, or press **Edit**, to
correct the sentence (each sentence stays separately editable and removable). Playback speeds run from 0.5× to 2×, there is
a volume control (it remembers your level), and the bar at the bottom shows who spoke when; click it to jump. The
**Verbatim / Clean** switch previews what a clean-verbatim export will look like. **Mark as reviewed** (top right) updates
the Transcripts screen.

For video recordings the video sits on the side by default. **Theater mode** (button under the video) makes it full width and
pins it to the top while the transcript scrolls below it, so you can watch what was on the screen while you read; the
video keeps playing when you switch, and the app remembers your choice. There is also a full-screen button.

Word highlighting uses Whisper's word timings. After you edit a sentence, or in the Clean view, the words are spread evenly
across that sentence instead, so the highlight stays close but is no longer exact.

> **Your original transcript is never modified.** Corrections are stored separately (`edits.json`) and cleanup
> (removing "um", stutters) is applied only when you export, so you can always export either version.

### Export formats

| Format | Use it for |
|---|---|
| **Word table (NVivo)** | A `Timespan \| Speaker \| Content` table, the layout NVivo's transcript importer maps columns from. |
| **Word document** | `[timestamp] Speaker: text` paragraphs (MAXQDA, ATLAS.ti, reading). |
| **WebVTT** | Captions. Speaker style: `Name: text` or standard `<v Name>` voice tags. |
| **SRT** | Captions for video players, MAXQDA, ATLAS.ti. |
| **Plain text** | `[timestamp] Speaker: text`, or speaker-grouped without timestamps. |
| **CSV** | `start, end, speaker, text`: Dedoose, Excel, R, Python. |
| **JSON** | Everything, including word-level timings and the settings used (for your methods section). |
| **PDF / Markdown** | Reading, printing, notes. |

Open the **Export** dialog from a file's row, from the review page, or with **Export all** on the project page. Tick the
formats you need (they are grouped by the software they are for), choose Verbatim or Clean text and whether to include
timestamps and whether to join each speaker's consecutive lines into one block (captions and CSV always keep timed
lines), then download. One file in one format downloads directly; anything more arrives as a `.zip`.

> **Please verify QDA imports once.** These layouts follow the vendors' published import guidance
> ([NVivo](https://help-nv.qsrinternational.com/14/win/Content/files/import-audio-video-transcripts.htm),
> [MAXQDA](https://www.maxqda.com/help-mx22/import/transcripts-with-timestamps),
> [ATLAS.ti](https://doc.atlasti.com/ManualMac.v9/Transcription/ImportingAutoTranscripts.html)), but they have not been
> tested inside those programs. Import one exported file into each tool you use and check that speakers, timestamps and
> media sync come through. Importers change between versions, and small tweaks (timestamp precision, speaker style) are
> easy to make in `src/transcriber/exporters/`.

PDF note: Latin-script languages (English, Spanish, Portuguese, French, German...) work everywhere; for other scripts the
PDF uses a system Unicode font if one is found.

## Command line

```bash
transcriber run interview.mp4 --formats vtt,docx_table,txt --model-size small --num-speakers 2 --clean
transcriber run interview.mp4 --no-diarize                       # no Hugging Face token needed
```

The original `whisper_diarize_clean_export.py` still works and writes a `.vtt` into `Local - Transcripts to revise/`.

## What changed from the original script

The review of the original script found several problems that affected transcript quality; the new core fixes them.

- **Speaker labels are assigned per word**, not per Whisper segment, so a segment that spans a speaker change is split
  correctly instead of attributing both speakers' words to one.
- **No more data-destroying cleanup.** "like", "you know", "I mean", "kind of" were removed everywhere ("I like pizza"
  became "I pizza"), `(inaudible)` markers were deleted, and an English spell-checker was applied to every language.
  Now only "um/uh/erm" and stutters are removed, only at export, and phrase fillers / pause markers / spell-check are
  opt-in (spell-check English only).
- **Failures no longer waste work.** Transcription and speaker results are saved as they finish.
- **Apple Silicon GPU** (mlx-whisper): about 5x faster than CPU on the same file in our test (`small` model).
  pyannote now uses the GPU when available (it previously always ran on CPU).
- **No files written next to your recordings**; the temporary WAV lives in the data folder.
- Dropped: `--punct` (it ran the whole audio through a punctuation model once per segment, and Whisper already
  punctuates) and the `--device/--compute-type/--auto-gpu` flags (now automatic).
- Segments are capped in length so captions and QDA imports stay readable.

## Where things live

Default data folder: `~/Transcriber` (override with the `TRANSCRIBER_DATA` environment variable).

```
app.db                                  projects, files, jobs
secrets/hf_token                        your Hugging Face token (mode 600)
projects/<project>/media/<file>/
    original.<ext>                      your upload, never modified
    audio.wav                           16 kHz working copy (safe to delete)
    checkpoints/                        saved intermediate results
    transcript.json                     the verbatim transcript with word timings
    edits.json                          your corrections and speaker names
```

The server only accepts connections from this computer (`127.0.0.1`), rejects requests from other web sites, and loads no
resources from the internet.

## Troubleshooting

- **"Hugging Face denied access"**: do the one-time setup above; *Test access* on the Setup page tells you which agreement is missing.
- **A file failed**: the error is shown on the row. **Retry** keeps whatever finished.
- **Speaker identification is fast on a Mac**: it runs on the Apple GPU (about 18× faster than the CPU in our test, 5 minutes of audio in ~10 s).
- **Very slow on a Mac**: run `transcriber doctor`; if the engine says `faster-whisper`, install the `mlx` extra.
- **Out of memory on long files**: use a smaller model.

## Development

```bash
uv pip install -e ".[dev]"
pytest
```

Layout: `src/transcriber/core` (pipeline, engines, alignment, text transforms), `exporters/`, `server/` (FastAPI, Jinja +
HTMX/Alpine, SQLite, job worker), `tests/`. Dependency pins in `pyproject.toml` are deliberate: pyannote.audio 3.1.1
predates NumPy 2 and huggingface-hub 1.x, torchaudio must match torch, and matplotlib is needed but not declared by
pyannote. PyTorch 2.6 refuses to load pyannote's checkpoints by default, so `core/diarize.py` allowlists the four classes
they contain (instead of turning the safety check off).
