# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `yt/`: YouTube video analysis. Downloads a video, picks keyframes with OpenCV, asks a vision-language model (Qwen2-VL or OpenAI) to describe them, and stores the results in SQLite. `query_video.py` searches the database.
- `package/soccer_player_recognition/`: draft package for player detection, segmentation, identification and classification, with demos, configs and a test suite.
- Demo GIF of the keyframe step on a CC BY-SA clip.
- CI on push and pull request: ruff error checks, a yt smoke check, the package tests.
- `yt/.env.example`, CONTRIBUTING.md, issue and PR templates.

### Changed

- The package imports and pytest runs. Undefined names, import cycles, device fallbacks and demo output paths fixed.
- Scripts that print synthetic results now say so. Hardcoded performance claims removed.
- `yt/`: requirements parse with pip, frame-diff and scene grouping fixed, first frame kept, Qwen output decoding and OpenAI local images fixed.
- README rewritten to say what runs and what is simulated.
- `.gitignore` covers every `.env*` file except `.env.example`.
- `yt/requirements.txt` uses `opencv-python-headless`, so the tool imports on servers without libGL.

### Removed

- Unused `requests` import in `yt/analyze_youtube_vlm.py`, which `yt/requirements.txt` did not install.
