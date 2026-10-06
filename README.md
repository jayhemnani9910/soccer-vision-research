[![Project page](https://img.shields.io/badge/Project%20page-Open-2ea44f?style=for-the-badge)](https://jayhemnani9910.github.io/soccer-vision-research/)

# soccer-vision-research

Research code for soccer video analysis. Two separate parts, at very different stages.
The project page above just renders this README.

## What is here

- `package/soccer_player_recognition/` - a soccer player recognition package (detection,
  segmentation, identification, classification). It is largely scaffolded: many modules,
  demos and docs, but no trained weights are included, and the RF-DETR and SAM2 classes are
  hand-written PyTorch models with untrained heads, not the official releases.
- `yt/` - a separate tool that downloads a YouTube video, picks keyframes, and asks a
  vision-language model (Qwen2-VL locally, or OpenAI) to describe them, storing results in
  SQLite. `yt/query_video.py` queries that database. See `yt/README.md`.

## What actually runs

Checked on CPU-only PyTorch, Python 3.11:

- The package imports: `soccer_player_recognition`, `models`, `utils`, `config`, `demos`
  and `UI`, and `PlayerRecognizer(enable_all_models=False)` constructs.
- `utils/` and `config/` helpers (drawing, image and video preprocessing, performance
  monitor, config loading/validation) run.
- Most demos under `demos/` run headless to the end. `sam2_demo` (tracking, occlusion) and
  `rf_detr_demo` still stop inside the untrained toy RF-DETR/SAM2 models.
- The test suite starts and collects 97 tests: 34 pass, 62 fail, 1 skipped (one memory
  test is flaky). Most failures are tests written against APIs the code does not have
  (e.g. `ImageUtils`, `DrawUtils`, `register_model(config=...)`), so they test code that
  was never written.
- `yt/query_video.py --help` runs. The full `yt/analyze_youtube_vlm.py` pipeline needs
  `yt-dlp`, a VLM backend and network access, and was not run end to end.

## Toy or simulated parts

- `simple_youtube_processor.py`, `process_youtube_video.py`, `standalone_complete_demo.py`,
  `standalone_single_demo.py` and most scripts under `demos/` use random or synthetic data in place of model output.
  Their numbers and drawn boxes are not real detections.
- Any "accuracy" or "FPS" printed by the demos comes from that demo data.

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r package/soccer_player_recognition/requirements.txt
pip install pytest psutil

cd package/soccer_player_recognition
python -m pytest tests -q -p no:cacheprovider
```

For the YouTube tool: `pip install -r yt/requirements.txt`, then follow `yt/README.md`.

## More docs

`package/soccer_player_recognition/README.md`, `DEMO_README.md` and `CONFIG_README.md`
describe the intended design. Treat them as plans, not as a description of working features.
