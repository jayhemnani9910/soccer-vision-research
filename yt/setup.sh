#!/bin/bash

echo "Setting up YouTube Video Analysis Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ required. Found Python $python_version"
    exit 1
fi

echo "✓ Python version OK: $python_version"

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create output directories
echo "Creating output directories..."
mkdir -p vlm_outputs/frames

# Set up environment variables template
echo "Creating environment template..."
cat > .env.template << 'EOF'
# VLM Backend Selection (qwen2vl or openai)
VLM_BACKEND=qwen2vl

# Qwen2-VL Configuration
QWEN_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Scene Detection Parameters
SCENE_CHANGE_METHOD=content-diff
CONTENT_DIFF_THRESHOLD=0.05
FALLBACK_EVERY_N_SECONDS=15
EOF

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.template to .env and configure your VLM backend"
echo "2. Source your environment: source .env"
echo "3. Run analysis: python analyze_youtube_vlm.py 'https://youtube.com/watch?v=VIDEO_ID'"
echo ""
echo "For OpenAI backend:"
echo "  export VLM_BACKEND=openai"
echo "  export OPENAI_API_KEY='your-api-key'"
echo ""
echo "For Qwen2-VL backend:"
echo "  export VLM_BACKEND=qwen2vl"
echo "  # The model will be downloaded automatically on first run"