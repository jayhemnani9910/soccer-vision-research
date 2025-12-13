#!/usr/bin/env python3
"""
Demo Launcher for Soccer Player Recognition System

This script provides a simple way to launch all the demo applications.
It handles dependencies and provides a menu interface for running demos.

Usage:
    python run_demos.py
    
Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'numpy',
        'opencv-python',
        'matplotlib',
        'Pillow',
        'tkinter',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'Pillow':
                import PIL
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def get_demo_files():
    """Get all available demo files."""
    demos_dir = Path(__file__).parent / "demos"
    ui_dir = Path(__file__).parent / "UI"
    
    demos = []
    
    # Find demo scripts
    demo_files = [
        ("complete_system_demo", "Complete System Demo", demos_dir / "complete_system_demo.py"),
        ("single_model_demo", "Single Model Demo", demos_dir / "single_model_demo.py"),
        ("real_time_demo", "Real-Time Processing Demo", demos_dir / "real_time_demo.py"),
        ("benchmark_demo", "Benchmark Demo", demos_dir / "benchmark_demo.py"),
        ("gui_interface", "GUI Interface", ui_dir / "demo_interface.py")
    ]
    
    for demo_id, description, file_path in demo_files:
        if file_path.exists():
            demos.append((demo_id, description, file_path))
        else:
            print(f"Warning: {file_path} not found")
    
    return demos

def run_demo(demo_file):
    """Run a specific demo."""
    try:
        print(f"\\n🚀 Launching {demo_file.name}...")
        print("=" * 60)
        
        # Run the demo
        result = subprocess.run([sys.executable, str(demo_file)], check=True)
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def interactive_menu():
    """Show interactive menu for demo selection."""
    print("\\n🏈 Soccer Player Recognition System - Demo Launcher")
    print("=" * 60)
    
    demos = get_demo_files()
    
    if not demos:
        print("❌ No demo files found!")
        return
    
    print("\\nAvailable Demos:")
    for i, (demo_id, description, _) in enumerate(demos, 1):
        print(f"  {i}. {description} ({demo_id})")
    
    print("\\n  0. Run all demos")
    print("  q. Quit")
    
    while True:
        try:
            choice = input("\\nSelect demo to run (0-{}, q): ".format(len(demos))).strip().lower()
            
            if choice == 'q':
                print("\\n👋 Goodbye!")
                break
            
            if choice == '0':
                # Run all demos
                print("\\n🚀 Running all demos...")
                success_count = 0
                
                for demo_id, description, demo_file in demos:
                    print(f"\\n--- Running {description} ---")
                    if run_demo(demo_file):
                        success_count += 1
                        print(f"✅ {description} completed successfully")
                    else:
                        print(f"❌ {description} failed")
                
                print(f"\\n🎯 Completed {success_count}/{len(demos)} demos successfully")
                break
            
            # Convert to integer and validate
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(demos):
                    demo_id, description, demo_file = demos[choice_num - 1]
                    if run_demo(demo_file):
                        print(f"\\n✅ {description} completed successfully")
                    else:
                        print(f"\\n❌ {description} failed")
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q'.")
                
        except KeyboardInterrupt:
            print("\\n\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Soccer Player Recognition System - Demo Launcher")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--demo", type=str, help="Run specific demo by name")
    parser.add_argument("--list", action="store_true", help="List available demos")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface directly")
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        print("\\nPlease install missing dependencies and try again.")
        return 1
    
    if args.check_deps:
        print("✅ All dependencies are satisfied!")
        return 0
    
    # List demos
    if args.list:
        demos = get_demo_files()
        print("\\nAvailable demos:")
        for demo_id, description, _ in demos:
            print(f"  • {demo_id}: {description}")
        return 0
    
    # Run specific demo
    if args.demo:
        demos = {demo_id: file_path for demo_id, _, file_path in get_demo_files()}
        
        if args.demo in demos:
            success = run_demo(demos[args.demo])
            return 0 if success else 1
        else:
            print(f"❌ Demo '{args.demo}' not found.")
            print(f"Available demos: {list(demos.keys())}")
            return 1
    
    # Launch GUI directly
    if args.gui:
        ui_file = Path(__file__).parent / "UI" / "demo_interface.py"
        if ui_file.exists():
            return subprocess.call([sys.executable, str(ui_file)])
        else:
            print("❌ GUI interface not found!")
            return 1
    
    # Interactive menu (default)
    interactive_menu()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())