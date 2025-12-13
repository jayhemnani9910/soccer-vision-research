#!/usr/bin/env python3
"""
Demo script showing how to use the YouTube video analysis system.
This example uses a short educational video for demonstration.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_demo():
    """Run a complete demo analysis."""
    
    # Example YouTube URL (short educational video)
    demo_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # You can replace with any appropriate video
    
    print("🎬 YouTube Video Analysis Demo")
    print("=" * 50)
    
    # Check if environment is set up
    vlm_backend = os.environ.get("VLM_BACKEND", "qwen2vl")
    print(f"VLM Backend: {vlm_backend}")
    
    if vlm_backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY not set!")
            print("Set it with: export OPENAI_API_KEY='your-key'")
            return False
    else:
        print("🤖 Using Qwen2-VL (will download model on first run)")
    
    print(f"\n📥 Analyzing video: {demo_url}")
    print("This will:")
    print("  1. Download the video")
    print("  2. Extract keyframes based on scene changes")
    print("  3. Run VLM agents on each keyframe")
    print("  4. Store results in SQLite database")
    print("  5. Generate scene narratives")
    
    # Ask for confirmation
    response = input("\nContinue? (y/N): ").strip().lower()
    if response != 'y':
        print("Demo cancelled.")
        return False
    
    try:
        # Run the analysis
        print("\n🚀 Starting analysis...")
        result = subprocess.run([
            sys.executable, "analyze_youtube_vlm.py", demo_url
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
            print("\n📊 Results:")
            print(result.stdout)
            
            # Show how to query
            print("\n🔍 Now you can query the results:")
            print("  python query_video.py scenes                    # List all scenes")
            print("  python query_video.py objects 'person'          # Find person objects")
            print("  python query_video.py text 'example'            # Find text content")
            print("  python query_video.py scene 0                  # Get scene 0 details")
            
        else:
            print("❌ Analysis failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def show_query_examples():
    """Show examples of querying the database."""
    print("\n📋 Query Examples:")
    print("=" * 30)
    
    examples = [
        ("List all scenes", "python query_video.py scenes"),
        ("Find person objects", "python query_video.py objects 'person'"),
        ("Find text content", "python query_video.py text 'button'"),
        ("Find movement events", "python query_video.py events 'movement'"),
        ("Get scene details", "python query_video.py scene 0"),
        ("Limit results", "python query_video.py objects 'car' --limit 5"),
    ]
    
    for desc, cmd in examples:
        print(f"\n{desc}:")
        print(f"  {cmd}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        show_query_examples()
    else:
        success = run_demo()
        if success:
            show_query_examples()