# YouTube Video Analysis with Vision-Language Models

This system provides comprehensive video analysis by extracting keyframes from YouTube videos and analyzing them with specialized VLM "agents" for scanning, tracking, and Q&A.

## Features

- **YouTube Video Download**: Automatically downloads best quality video using yt-dlp
- **Smart Scene Detection**: Detects scene changes and samples keyframes intelligently
- **Three VLM Agents**:
  - **Scanner**: Detailed description of frame contents (objects, text, UI, etc.)
  - **Tracker**: Tracks changes between consecutive frames
  - **Q&A**: Generates disambiguation questions for each frame
- **SQLite Storage**: Structured database for analytics and querying
- **Query Interface**: Command-line tool to search analyzed video content

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Choose your VLM backend**:

**For Qwen2-VL (recommended, local)**:
```bash
export VLM_BACKEND=qwen2vl
export QWEN_MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"  # or "Qwen/Qwen2-VL-7B-Instruct"
```

**For OpenAI GPT-4o**:
```bash
export VLM_BACKEND=openai
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # or "gpt-4o"
```

## Usage

### Analyze a YouTube Video

```bash
python analyze_youtube_vlm.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The system will:
1. Download the video
2. Extract keyframes based on scene changes
3. Run three VLM agents on each keyframe
4. Store results in SQLite database
5. Generate scene narratives

### Query the Analysis

```bash
# Find objects by name
python query_video.py objects "person" --limit 5

# Find frames containing specific text
python query_video.py text "login" --limit 3

# Find events by type
python query_video.py events "movement" --limit 10

# Get detailed scene information
python query_video.py scene 5

# List all scenes
python query_video.py scenes
```

## Output Structure

```
vlm_outputs/
├── video_analytics.db          # SQLite database
├── frames/                     # Extracted keyframes
│   ├── scene_0000_frame_000000.png
│   ├── scene_0001_frame_000123.png
│   └── ...
└── scene_narratives.jsonl      # Scene-by-scene summaries
```

## Database Schema

- **scenes**: Scene boundaries and timing
- **frames**: Frame metadata and VLM outputs
- **entities**: Object tracking across scenes
- **events**: Detected events and changes
- **qa_questions**: Generated questions for disambiguation

## Configuration

Key parameters in `analyze_youtube_vlm.py`:

- `SCENE_CHANGE_METHOD`: "content-diff" or "hist-threshold"
- `CONTENT_DIFF_THRESHOLD`: Sensitivity for scene detection (0.05)
- `FALLBACK_EVERY_N_SECONDS`: Force frame every N seconds (15)
- `VLM_BACKEND`: "qwen2vl" or "openai"

## VLM Agent Details

### Scanner Agent
Analyzes each frame and returns:
- Objects with attributes and bounding boxes
- Text content and locations
- UI elements
- Scene summary (<80 words)
- Notable details
- Quality assessment
- Object counts

### Tracker Agent
Compares consecutive frames and reports:
- Newly introduced objects
- Removed objects
- Object movements
- Attribute changes
- Text changes
- Global events

### Q&A Agent
Generates 2-3 questions per frame:
- Object identity questions
- Count questions
- Text content questions
- State questions
- Other disambiguation questions

## Performance Tips

- **GPU Usage**: Qwen2-VL runs much faster with CUDA GPU
- **Model Choice**: 2B model is faster, 7B model is more accurate
- **Sampling**: Adjust `FALLBACK_EVERY_N_SECONDS` for more/less dense analysis
- **Thresholds**: Tune `CONTENT_DIFF_THRESHOLD` for scene detection sensitivity

## Advanced Usage

### Environment Variables

```bash
# VLM Backend Selection
export VLM_BACKEND=qwen2vl  # or openai

# Qwen2-VL Configuration
export QWEN_MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"

# OpenAI Configuration
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o"

# Scene Detection
export SCENE_CHANGE_METHOD="content-diff"
export CONTENT_DIFF_THRESHOLD=0.05
export FALLBACK_EVERY_N_SECONDS=15
```

### Custom Prompts

You can modify the agent prompts (`SCANNER_PROMPT`, `TRACKER_PROMPT`, `QA_PROMPT`) in the code to focus on specific aspects of your videos.

## Troubleshooting

1. **Memory Issues**: Use smaller model (2B instead of 7B) or increase fallback interval
2. **Slow Performance**: Ensure GPU is available for Qwen2-VL
3. **Missing Frames**: Check video format and scene detection thresholds
4. **VLM Errors**: Verify API keys and model access

## Extensions

The system can be extended with:
- Web interface for querying
- Real-time video analysis
- Additional VLM backends
- Advanced entity tracking across scenes
- Semantic search with embeddings
- Video summarization and highlights generation