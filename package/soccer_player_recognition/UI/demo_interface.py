#!/usr/bin/env python3
"""
GUI Demo Interface for Soccer Player Recognition

This is a comprehensive GUI application for testing and demonstrating
the soccer player recognition system with an intuitive interface.

Features:
- Interactive model testing
- Real-time processing controls
- Performance monitoring
- File browser and visualization
- Configuration management

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import sys
import os
import json
import time
import threading
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try importing demo modules
try:
    from demos.single_model_demo import SingleModelDemo
    from demos.real_time_demo import RealTimeDemo, RealTimeProcessor
    from demos.benchmark_demo import BenchmarkDemo
    from demos.complete_system_demo import CompleteSystemDemo
except ImportError as e:
    print(f"Warning: Could not import demo modules: {e}")
    # Create mock classes for demonstration
    class MockDemo:
        def __init__(self):
            pass
        
        def run_demo(self):
            messagebox.showinfo("Demo", "Demo functionality not available")
    
    SingleModelDemo = MockDemo
    RealTimeDemo = MockDemo
    BenchmarkDemo = MockDemo
    CompleteSystemDemo = MockDemo


class ImageViewer:
    """Image viewer widget with zoom and pan capabilities."""
    
    def __init__(self, parent):
        """Initialize image viewer."""
        self.parent = parent
        self.image = None
        self.tk_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Create canvas
        self.canvas = tk.Canvas(parent, bg='gray', cursor='hand2')
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bind events
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<MouseWheel>', self.on_scroll)
    
    def load_image(self, image_path: str):
        """Load and display an image."""
        try:
            # Load image with PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            self.image = pil_image
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def display_image(self):
        """Display the current image."""
        if not self.image:
            return
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, schedule display
            self.parent.after(100, self.display_image)
            return
        
        # Calculate display size
        img_width, img_height = self.image.size
        display_width = int(img_width * self.scale)
        display_height = int(img_height * self.scale)
        
        # Resize image
        resized_image = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y,
            anchor='nw', image=self.tk_image
        )
    
    def on_click(self, event):
        """Handle mouse click."""
        self.last_x = event.x
        self.last_y = event.y
    
    def on_drag(self, event):
        """Handle mouse drag for panning."""
        if hasattr(self, 'last_x'):
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.offset_x += dx
            self.offset_y += dy
            self.display_image()
            self.last_x = event.x
            self.last_y = event.y
    
    def on_scroll(self, event):
        """Handle mouse wheel for zooming."""
        if not self.image:
            return
        
        # Zoom factor
        if event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale *= 0.9
        
        # Clamp scale
        self.scale = max(0.1, min(5.0, self.scale))
        
        self.display_image()


class ModelControlPanel:
    """Control panel for individual model testing."""
    
    def __init__(self, parent, demo_callback):
        """Initialize model control panel."""
        self.parent = parent
        self.demo_callback = demo_callback
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Model Controls", padding=10)
        self.frame.pack(fill='x', padx=10, pady=5)
        
        # Model selection
        ttk.Label(self.frame, text="Select Model:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        self.model_var = tk.StringVar(value="rf_detr")
        self.model_combo = ttk.Combobox(
            self.frame, 
            textvariable=self.model_var,
            values=["rf_detr", "sam2", "siglip", "resnet"],
            state="readonly"
        )
        self.model_combo.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        
        # Input size selection
        ttk.Label(self.frame, text="Input Size:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        
        self.size_var = tk.StringVar(value="640x640")
        self.size_combo = ttk.Combobox(
            self.frame,
            textvariable=self.size_var,
            values=["224x224", "384x384", "512x512", "640x640", "1024x1024"],
            state="readonly"
        )
        self.size_combo.grid(row=0, column=3, sticky='ew', padx=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Run Test", command=self.run_test).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side='left', padx=5)
        
        # Image path
        self.image_path_var = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.image_path_var, state="readonly").grid(
            row=2, column=0, columnspan=4, sticky='ew', pady=5
        )
        
        # Configure grid weights
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(3, weight=1)
    
    def load_image(self):
        """Load image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path_var.set(file_path)
    
    def run_test(self):
        """Run model test."""
        if not self.image_path_var.get():
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        # Get configuration
        model_name = self.model_var.get()
        input_size = self.size_var.get()
        
        # Run test in a thread to avoid UI freezing
        def test_thread():
            try:
                # Simulate test execution
                time.sleep(2)  # Simulate processing time
                
                # Create mock results
                results = {
                    'model': model_name,
                    'input_size': input_size,
                    'processing_time': np.random.uniform(0.1, 0.5),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'detections': [
                        {
                            'bbox': [100, 100, 200, 200],
                            'class': 'player',
                            'confidence': np.random.uniform(0.8, 0.95)
                        }
                    ]
                }
                
                # Update UI
                self.parent.after(0, lambda: self.show_results(results))
                
            except Exception as e:
                self.parent.after(0, lambda: messagebox.showerror("Error", f"Test failed: {e}"))
        
        thread = threading.Thread(target=test_thread)
        thread.daemon = True
        thread.start()
    
    def show_results(self, results):
        """Display test results."""
        messagebox.showinfo("Test Results", 
                           f"Model: {results['model']}\\n"
                           f"Processing Time: {results['processing_time']:.3f}s\\n"
                           f"Confidence: {results['confidence']:.2%}\\n"
                           f"Detections: {len(results['detections'])}")
    
    def clear_results(self):
        """Clear test results."""
        self.image_path_var.set("")


class PerformanceMonitor:
    """Real-time performance monitoring widget."""
    
    def __init__(self, parent):
        """Initialize performance monitor."""
        self.parent = parent
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'fps': []
        }
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Performance Monitor", padding=10)
        self.frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring", command=self.stop_monitoring, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Clear Data", command=self.clear_data).pack(side='left', padx=5)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Statistics labels
        stats_frame = ttk.Frame(self.frame)
        stats_frame.pack(fill='x', pady=(10, 0))
        
        self.cpu_label = ttk.Label(stats_frame, text="CPU: 0%")
        self.cpu_label.pack(side='left', padx=10)
        
        self.memory_label = ttk.Label(stats_frame, text="Memory: 0%")
        self.memory_label.pack(side='left', padx=10)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0.0")
        self.fps_label.pack(side='left', padx=10)
        
        # Start update timer
        self.update_timer = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Start update loop
        self.update_monitor()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if self.update_timer:
            self.parent.after_cancel(self.update_timer)
    
    def update_monitor(self):
        """Update monitoring data and display."""
        if not self.monitoring:
            return
        
        # Simulate performance data
        cpu_usage = np.random.uniform(10, 80)
        memory_usage = np.random.uniform(30, 70)
        fps = np.random.uniform(20, 60)
        
        # Add to data
        current_time = time.time()
        self.data['timestamps'].append(current_time)
        self.data['cpu_usage'].append(cpu_usage)
        self.data['memory_usage'].append(memory_usage)
        self.data['fps'].append(fps)
        
        # Keep only last 50 data points
        for key in self.data:
            if len(self.data[key]) > 50:
                self.data[key] = self.data[key][-50:]
        
        # Update plot
        self.update_plot()
        
        # Update statistics
        self.cpu_label.config(text=f"CPU: {cpu_usage:.1f}%")
        self.memory_label.config(text=f"Memory: {memory_usage:.1f}%")
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        
        # Schedule next update
        self.update_timer = self.parent.after(100, self.update_monitor)
    
    def update_plot(self):
        """Update performance plot."""
        self.subplot.clear()
        
        # Plot CPU and Memory usage
        timestamps = list(range(len(self.data['cpu_usage'])))
        
        self.subplot.plot(timestamps, self.data['cpu_usage'], 'b-', label='CPU %', linewidth=2)
        self.subplot.plot(timestamps, self.data['memory_usage'], 'r-', label='Memory %', linewidth=2)
        self.subplot.plot(timestamps, self.data['fps'], 'g-', label='FPS', linewidth=2)
        
        self.subplot.set_xlabel('Time (samples)')
        self.subplot.set_ylabel('Usage % / FPS')
        self.subplot.set_title('System Performance')
        self.subplot.legend()
        self.subplot.grid(True, alpha=0.3)
        
        # Update canvas
        self.canvas.draw()
    
    def clear_data(self):
        """Clear monitoring data."""
        for key in self.data:
            self.data[key] = []


class DemoControlPanel:
    """Control panel for running different demos."""
    
    def __init__(self, parent):
        """Initialize demo control panel."""
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Demo Controls", padding=10)
        self.frame.pack(fill='x', padx=10, pady=5)
        
        # Demo selection
        ttk.Label(self.frame, text="Select Demo:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        self.demo_var = tk.StringVar(value="single_model")
        self.demo_combo = ttk.Combobox(
            self.frame,
            textvariable=self.demo_var,
            values=[
                "single_model", 
                "complete_system", 
                "real_time", 
                "benchmark"
            ],
            state="readonly"
        )
        self.demo_combo.grid(row=0, column=1, sticky='ew', padx=(0, 20))
        
        # Configuration options
        self.iterations_var = tk.StringVar(value="50")
        ttk.Label(self.frame, text="Iterations:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        ttk.Entry(self.frame, textvariable=self.iterations_var, width=10).grid(row=0, column=3, sticky='w', padx=(0, 20))
        
        # Control buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Run Demo", command=self.run_demo).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop Demo", command=self.stop_demo).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side='left', padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=4, sticky='ew', pady=5)
        
        # Status text
        self.status_text = scrolledtext.ScrolledText(self.frame, height=8, wrap=tk.WORD)
        self.status_text.grid(row=3, column=0, columnspan=4, sticky='nsew', pady=5)
        
        # Configure grid weights
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_rowconfigure(3, weight=1)
        
        # Demo running state
        self.demo_running = False
        self.demo_thread = None
    
    def log_message(self, message: str):
        """Add message to status log."""
        self.status_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\\n")
        self.status_text.see(tk.END)
        self.parent.update_idletasks()
    
    def run_demo(self):
        """Run selected demo."""
        if self.demo_running:
            messagebox.showwarning("Warning", "A demo is already running")
            return
        
        demo_type = self.demo_var.get()
        iterations = int(self.iterations_var.get())
        
        self.log_message(f"Starting {demo_type} demo with {iterations} iterations...")
        
        # Start demo thread
        def demo_thread():
            try:
                self.demo_running = True
                
                # Simulate demo execution
                for i in range(iterations):
                    if not self.demo_running:
                        break
                    
                    # Simulate progress
                    self.parent.after(0, lambda i=i: self.progress.configure(value=i))
                    
                    # Log progress
                    self.log_message(f"Processing iteration {i+1}/{iterations}")
                    
                    # Simulate work
                    time.sleep(0.1)
                
                # Final result
                self.parent.after(0, lambda: self.log_message("Demo completed successfully!"))
                
            except Exception as e:
                self.parent.after(0, lambda: self.log_message(f"Demo failed: {e}"))
            finally:
                self.demo_running = False
                self.parent.after(0, lambda: self.progress.configure(value=0))
        
        self.demo_thread = threading.Thread(target=demo_thread)
        self.demo_thread.daemon = True
        self.demo_thread.start()
    
    def stop_demo(self):
        """Stop running demo."""
        if self.demo_running:
            self.demo_running = False
            self.log_message("Stopping demo...")
    
    def save_results(self):
        """Save demo results."""
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                results = {
                    'timestamp': time.time(),
                    'demo_type': self.demo_var.get(),
                    'iterations': int(self.iterations_var.get()),
                    'status': 'completed',
                    'log': self.status_text.get(1.0, tk.END)
                }
                
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {e}")


class FileManager:
    """File management widget for loading images and videos."""
    
    def __init__(self, parent, callback):
        """Initialize file manager."""
        self.parent = parent
        self.callback = callback
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="File Manager", padding=10)
        self.frame.pack(fill='x', padx=10, pady=5)
        
        # File path
        ttk.Label(self.frame, text="File:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(self.frame, textvariable=self.file_path_var, state="readonly")
        self.file_entry.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        
        ttk.Button(self.frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(self.frame, text="Load", command=self.load_file).grid(row=0, column=3)
        
        # File type filter
        filter_frame = ttk.Frame(self.frame)
        filter_frame.grid(row=1, column=0, columnspan=4, pady=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side='left')
        
        self.filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            values=["all", "images", "videos", "text"],
            state="readonly",
            width=15
        )
        filter_combo.pack(side='left', padx=(10, 0))
        
        # Configure grid
        self.frame.grid_columnconfigure(1, weight=1)
    
    def browse_file(self):
        """Browse for file."""
        filter_type = self.filter_var.get()
        
        if filter_type == "all":
            filetypes = [("All files", "*.*")]
        elif filter_type == "images":
            filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        elif filter_type == "videos":
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        else:
            filetypes = [("Text files", "*.txt *.json *.csv"), ("All files", "*.*")]
        
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=filetypes
        )
        
        if file_path:
            self.file_path_var.set(file_path)
    
    def load_file(self):
        """Load selected file."""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        if self.callback:
            self.callback(file_path)


class SettingsPanel:
    """Settings and configuration panel."""
    
    def __init__(self, parent):
        """Initialize settings panel."""
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Settings", padding=10)
        self.frame.pack(fill='x', padx=10, pady=5)
        
        # Model settings
        ttk.Label(self.frame, text="Default Model:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        self.default_model_var = tk.StringVar(value="rf_detr")
        model_combo = ttk.Combobox(
            self.frame,
            textvariable=self.default_model_var,
            values=["rf_detr", "sam2", "siglip", "resnet"],
            state="readonly"
        )
        model_combo.grid(row=0, column=1, sticky='ew', padx=(0, 20))
        
        # Processing settings
        ttk.Label(self.frame, text="Processing Threads:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        
        self.threads_var = tk.StringVar(value="4")
        ttk.Entry(self.frame, textvariable=self.threads_var, width=10).grid(row=0, column=3, sticky='w')
        
        # Output settings
        ttk.Label(self.frame, text="Output Directory:").grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        
        self.output_dir_var = tk.StringVar(value="outputs")
        ttk.Entry(self.frame, textvariable=self.output_dir_var).grid(row=1, column=1, columnspan=2, sticky='ew', padx=(0, 10), pady=(10, 0))
        
        ttk.Button(self.frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=3, pady=(10, 0))
        
        # Log level
        ttk.Label(self.frame, text="Log Level:").grid(row=2, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        
        self.log_level_var = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(
            self.frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly"
        )
        log_combo.grid(row=2, column=1, sticky='w', pady=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=20)
        
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Settings", command=self.load_settings).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset Defaults", command=self.reset_defaults).pack(side='left', padx=5)
        
        # Configure grid
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def save_settings(self):
        """Save current settings."""
        settings = {
            'default_model': self.default_model_var.get(),
            'processing_threads': int(self.threads_var.get()),
            'output_directory': self.output_dir_var.get(),
            'log_level': self.log_level_var.get()
        }
        
        settings_file = Path("gui_settings.json")
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        messagebox.showinfo("Success", f"Settings saved to {settings_file}")
    
    def load_settings(self):
        """Load settings from file."""
        settings_file = Path("gui_settings.json")
        
        if not settings_file.exists():
            messagebox.showwarning("Warning", "No settings file found")
            return
        
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            self.default_model_var.set(settings.get('default_model', 'rf_detr'))
            self.threads_var.set(str(settings.get('processing_threads', 4)))
            self.output_dir_var.set(settings.get('output_directory', 'outputs'))
            self.log_level_var.set(settings.get('log_level', 'INFO'))
            
            messagebox.showinfo("Success", "Settings loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load settings: {e}")
    
    def reset_defaults(self):
        """Reset to default settings."""
        self.default_model_var.set("rf_detr")
        self.threads_var.set("4")
        self.output_dir_var.set("outputs")
        self.log_level_var.set("INFO")
        
        messagebox.showinfo("Success", "Settings reset to defaults")


class SoccerRecognitionGUI:
    """Main GUI application for soccer player recognition system."""
    
    def __init__(self, root):
        """Initialize the main GUI application."""
        self.root = root
        self.root.title("Soccer Player Recognition System - Demo Interface")
        self.root.geometry("1200x800")
        
        # Initialize demo instances
        self.single_model_demo = SingleModelDemo()
        self.real_time_demo = RealTimeDemo()
        self.benchmark_demo = BenchmarkDemo()
        self.complete_system_demo = CompleteSystemDemo()
        
        # Create menu
        self.create_menu()
        
        # Create main interface
        self.create_main_interface()
        
        # Load settings
        self.load_default_settings()
    
    def create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Demo menu
        demo_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Demos", menu=demo_menu)
        demo_menu.add_command(label="Single Model Test", command=lambda: self.run_demo("single_model"))
        demo_menu.add_command(label="Complete System", command=lambda: self.run_demo("complete_system"))
        demo_menu.add_command(label="Real-Time Processing", command=lambda: self.run_demo("real_time"))
        demo_menu.add_command(label="Benchmark Suite", command=lambda: self.run_demo("benchmark"))
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Performance Monitor", command=self.show_performance_monitor)
        tools_menu.add_command(label="Settings", command=self.show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_interface(self):
        """Create the main interface."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_main_tab()
        self.create_model_tab()
        self.create_realtime_tab()
        self.create_benchmark_tab()
        self.create_settings_tab()
    
    def create_main_tab(self):
        """Create the main dashboard tab."""
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Dashboard")
        
        # Welcome message
        welcome_frame = ttk.Frame(main_frame)
        welcome_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Label(
            welcome_frame,
            text="Soccer Player Recognition System",
            font=('Arial', 24, 'bold')
        ).pack()
        
        ttk.Label(
            welcome_frame,
            text="Comprehensive Demo and Testing Interface",
            font=('Arial', 12)
        ).pack(pady=(5, 20))
        
        # Quick actions
        actions_frame = ttk.LabelFrame(main_frame, text="Quick Actions", padding=20)
        actions_frame.pack(fill='x', padx=20, pady=10)
        
        button_frame = ttk.Frame(actions_frame)
        button_frame.pack()
        
        ttk.Button(
            button_frame,
            text="Load Test Image",
            command=self.load_image,
            width=20
        ).pack(side='left', padx=10, pady=5)
        
        ttk.Button(
            button_frame,
            text="Run Single Model Test",
            command=lambda: self.run_demo("single_model"),
            width=20
        ).pack(side='left', padx=10, pady=5)
        
        ttk.Button(
            button_frame,
            text="Start Real-Time Demo",
            command=lambda: self.run_demo("real_time"),
            width=20
        ).pack(side='left', padx=10, pady=5)
        
        ttk.Button(
            button_frame,
            text="Run Benchmark",
            command=lambda: self.run_demo("benchmark"),
            width=20
        ).pack(side='left', padx=10, pady=5)
        
        # Image viewer
        viewer_frame = ttk.LabelFrame(main_frame, text="Image Viewer", padding=10)
        viewer_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.image_viewer = ImageViewer(viewer_frame)
        
        # File manager
        self.file_manager = FileManager(main_frame, self.on_file_load)
    
    def create_model_tab(self):
        """Create the model testing tab."""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="Model Testing")
        
        # Create paned window for better layout
        paned = ttk.PanedWindow(model_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel: Controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        self.model_control = ModelControlPanel(left_frame, self.on_model_result)
        
        # Right panel: Results and visualization
        right_frame = ttk.LabelFrame(paned, text="Results", padding=10)
        paned.add(right_frame, weight=2)
        
        # Create results text area
        self.results_text = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True)
    
    def create_realtime_tab(self):
        """Create the real-time processing tab."""
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="Real-Time Processing")
        
        # Create paned window
        paned = ttk.PanedWindow(realtime_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel: Controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Real-time controls
        control_frame = ttk.LabelFrame(left_frame, text="Real-Time Controls", padding=10)
        control_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(control_frame, text="Target FPS:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.target_fps_var = tk.StringVar(value="30")
        ttk.Entry(control_frame, textvariable=self.target_fps_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(control_frame, text="Resolution:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(
            control_frame,
            textvariable=self.resolution_var,
            values=["480p", "720p", "1080p"],
            state="readonly",
            width=10
        )
        resolution_combo.grid(row=0, column=3)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        self.start_rt_button = ttk.Button(button_frame, text="Start Stream", command=self.start_real_time)
        self.start_rt_button.pack(side='left', padx=5)
        
        self.stop_rt_button = ttk.Button(button_frame, text="Stop Stream", command=self.stop_real_time, state='disabled')
        self.stop_rt_button.pack(side='left', padx=5)
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor(left_frame)
        
        # Right panel: Stream visualization
        right_frame = ttk.LabelFrame(paned, text="Live Stream", padding=10)
        paned.add(right_frame, weight=2)
        
        # Create stream display (placeholder)
        stream_display = tk.Canvas(right_frame, bg='black', height=400)
        stream_display.pack(fill='both', expand=True)
        
        # Add placeholder text
        stream_display.create_text(
            200, 200, text="Real-time stream will appear here",
            fill='white', font=('Arial', 14)
        )
        
        # Stream info
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill='x', pady=5)
        
        self.stream_status_var = tk.StringVar(value="Stream: Stopped")
        ttk.Label(info_frame, textvariable=self.stream_status_var).pack(side='left')
    
    def create_benchmark_tab(self):
        """Create the benchmark testing tab."""
        benchmark_frame = ttk.Frame(self.notebook)
        self.notebook.add(benchmark_frame, text="Benchmark")
        
        # Create paned window
        paned = ttk.PanedWindow(benchmark_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel: Controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        self.demo_control = DemoControlPanel(left_frame)
        
        # Right panel: Results and charts
        right_frame = ttk.LabelFrame(paned, text="Benchmark Results", padding=10)
        paned.add(right_frame, weight=2)
        
        # Create notebook for different result views
        results_notebook = ttk.Notebook(right_frame)
        results_notebook.pack(fill='both', expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(results_notebook)
        results_notebook.add(summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=15)
        self.summary_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Charts tab
        charts_frame = ttk.Frame(results_notebook)
        results_notebook.add(charts_frame, text="Charts")
        
        # Create simple chart placeholder
        chart_canvas = tk.Canvas(charts_frame, bg='white', height=300)
        chart_canvas.pack(fill='x', padx=5, pady=5)
        chart_canvas.create_text(200, 150, text="Performance charts will appear here", fill='gray')
    
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        self.settings_panel = SettingsPanel(settings_frame)
    
    def load_default_settings(self):
        """Load default settings."""
        # This would load settings from a file in a real implementation
        pass
    
    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_viewer.load_image(file_path)
    
    def load_video(self):
        """Load a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            messagebox.showinfo("Info", f"Video loaded: {file_path}")
    
    def on_file_load(self, file_path: str):
        """Handle file loading."""
        self.image_viewer.load_image(file_path)
        
        # Switch to main tab
        self.notebook.select(0)
    
    def on_model_result(self, result: Dict[str, Any]):
        """Handle model test results."""
        # Display results in text area
        self.results_text.insert(tk.END, f"\\n--- Result at {time.strftime('%H:%M:%S')} ---\\n")
        self.results_text.insert(tk.END, json.dumps(result, indent=2, default=str))
        self.results_text.insert(tk.END, "\\n\\n")
        self.results_text.see(tk.END)
    
    def start_real_time(self):
        """Start real-time processing."""
        target_fps = int(self.target_fps_var.get())
        resolution = self.resolution_var.get()
        
        self.start_rt_button.config(state='disabled')
        self.stop_rt_button.config(state='normal')
        self.stream_status_var.set(f"Stream: Running at {target_fps} FPS ({resolution})")
        
        # Start real-time processing thread
        def rt_thread():
            # Simulate real-time processing
            for i in range(100):  # Run for 100 frames
                time.sleep(1.0 / target_fps)
                # Here would be the actual processing logic
        
        thread = threading.Thread(target=rt_thread)
        thread.daemon = True
        thread.start()
    
    def stop_real_time(self):
        """Stop real-time processing."""
        self.start_rt_button.config(state='normal')
        self.stop_rt_button.config(state='disabled')
        self.stream_status_var.set("Stream: Stopped")
    
    def run_demo(self, demo_type: str):
        """Run a specific demo."""
        # Switch to appropriate tab
        tab_mapping = {
            "single_model": 1,
            "complete_system": 1,
            "real_time": 2,
            "benchmark": 3
        }
        
        tab_index = tab_mapping.get(demo_type, 0)
        self.notebook.select(tab_index)
        
        # Run demo in appropriate control panel
        if demo_type == "benchmark":
            self.demo_control.run_demo()
        elif demo_type == "single_model":
            messagebox.showinfo("Info", "Use the Model Testing tab for single model tests")
        elif demo_type == "real_time":
            messagebox.showinfo("Info", "Use the Real-Time Processing tab for real-time demos")
    
    def show_performance_monitor(self):
        """Show performance monitor window."""
        # This would show a separate performance monitoring window
        messagebox.showinfo("Info", "Performance monitor is available in the Real-Time Processing tab")
    
    def show_settings(self):
        """Show settings dialog."""
        self.notebook.select(4)  # Switch to settings tab
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
Soccer Player Recognition System
Demo Interface v1.0

This application demonstrates comprehensive
soccer player recognition capabilities including:

• Object Detection (RF-DETR)
• Segmentation (SAM2)
• Player Identification (SigLIP)
• Classification (ResNet)
• Real-Time Processing
• Performance Benchmarking

Built with Python, OpenCV, and PyTorch
        """
        
        messagebox.showinfo("About", about_text)


def main():
    """Main function to start the GUI application."""
    # Create and run the application
    root = tk.Tk()
    app = SoccerRecognitionGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())