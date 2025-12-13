"""
Performance Monitor for Soccer Player Recognition System

Provides functionality for tracking and analyzing model performance metrics
including mAP, FPS, precision, recall, and other evaluation metrics.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import threading
from datetime import datetime
import csv

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Data class for storing metrics snapshots."""
    timestamp: float
    metric_name: str
    metric_value: float
    model_id: str
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceReport:
    """Data class for performance reports."""
    model_id: str
    total_inferences: int
    avg_inference_time: float
    fps: float
    memory_usage_mb: float
    gpu_utilization: Optional[float]
    cpu_utilization: float
    accuracy_metrics: Dict[str, float]
    custom_metrics: Dict[str, float]


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for ML models.
    
    Features:
    - Real-time performance tracking
    - Metrics calculation (mAP, FPS, precision, recall, etc.)
    - Memory and resource usage monitoring
    - Performance reporting and visualization
    - Thread-safe operations
    """
    
    def __init__(
        self, 
        output_dir: str = "outputs/performance",
        save_interval: int = 100,
        max_history: int = 10000
    ):
        """Initialize the performance monitor.
        
        Args:
            output_dir: Directory to save performance reports
            save_interval: Save metrics every N inferences
            max_history: Maximum number of metrics to keep in history
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.max_history = max_history
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.model_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance counters
        self.inference_counters: Dict[str, int] = defaultdict(int)
        self.total_inference_times: Dict[str, float] = defaultdict(float)
        
        # Resource monitoring
        self.resource_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
        logger.info(f"PerformanceMonitor initialized with output_dir: {self.output_dir}")
    
    def start_inference(self, model_id: str) -> float:
        """
        Start timing an inference.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Timestamp when inference started
        """
        with self._lock:
            self.inference_counters[model_id] += 1
            return time.time()
    
    def end_inference(
        self, 
        model_id: str, 
        start_time: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        End timing an inference and record metrics.
        
        Args:
            model_id: Unique identifier for the model
            start_time: Timestamp when inference started
            additional_data: Additional data to store with the metrics
            
        Returns:
            Inference duration in seconds
        """
        end_time = time.time()
        inference_time = end_time - start_time
        
        with self._lock:
            self.total_inference_times[model_id] += inference_time
            
            # Store inference time metrics
            snapshot = MetricsSnapshot(
                timestamp=end_time,
                metric_name="inference_time",
                metric_value=inference_time,
                model_id=model_id,
                additional_data=additional_data
            )
            self.metrics_history[f"{model_id}_inference_time"].append(snapshot)
            
            # Update model statistics
            if model_id not in self.model_stats:
                self.model_stats[model_id] = {}
            
            self.model_stats[model_id].update({
                'last_inference_time': inference_time,
                'total_inferences': self.inference_counters[model_id],
                'avg_inference_time': (
                    self.total_inference_times[model_id] / self.inference_counters[model_id]
                )
            })
            
            # Auto-save metrics periodically
            if self.inference_counters[model_id] % self.save_interval == 0:
                self._save_metrics_snapshot(snapshot)
        
        return inference_time
    
    def calculate_fps(self, model_id: str, time_window: float = 60.0) -> float:
        """
        Calculate frames per second for a model.
        
        Args:
            model_id: Unique identifier for the model
            time_window: Time window in seconds to calculate FPS over
            
        Returns:
            FPS value
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - time_window
            
            # Count inferences in time window
            recent_inferences = [
                snapshot for snapshot in self.metrics_history[f"{model_id}_inference_time"]
                if snapshot.timestamp >= window_start
            ]
            
            fps = len(recent_inferences) / time_window
            
            # Update model stats
            if model_id not in self.model_stats:
                self.model_stats[model_id] = {}
            self.model_stats[model_id]['fps'] = fps
            
            return fps
    
    def calculate_map(
        self, 
        predictions: List[Any], 
        ground_truth: List[Any],
        iou_threshold: float = 0.5,
        model_id: str = "unknown"
    ) -> float:
        """
        Calculate mean Average Precision (mAP) for object detection models.
        
        Args:
            predictions: List of prediction results
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching predictions to ground truth
            model_id: Unique identifier for the model
            
        Returns:
            mAP value
        """
        try:
            map_score = self._compute_map(predictions, ground_truth, iou_threshold)
            
            # Record mAP metric
            snapshot = MetricsSnapshot(
                timestamp=time.time(),
                metric_name="map",
                metric_value=map_score,
                model_id=model_id,
                additional_data={
                    "iou_threshold": iou_threshold,
                    "num_predictions": len(predictions),
                    "num_ground_truth": len(ground_truth)
                }
            )
            
            with self._lock:
                self.metrics_history[f"{model_id}_map"].append(snapshot)
                
                if model_id not in self.model_stats:
                    self.model_stats[model_id] = {}
                self.model_stats[model_id]['map'] = map_score
            
            return map_score
            
        except Exception as e:
            logger.error(f"Error calculating mAP: {e}")
            return 0.0
    
    def calculate_precision_recall(
        self,
        predictions: Union[np.ndarray, List],
        ground_truth: Union[np.ndarray, List],
        model_id: str = "unknown"
    ) -> Tuple[float, float]:
        """
        Calculate precision and recall for classification/detection models.
        
        Args:
            predictions: Model predictions (scores, classes, boxes, etc.)
            ground_truth: Ground truth labels/annotations
            model_id: Unique identifier for the model
            
        Returns:
            Tuple of (precision, recall)
        """
        try:
            if isinstance(predictions, np.ndarray) and isinstance(ground_truth, np.ndarray):
                # Binary classification case
                pred_binary = (predictions > 0.5).astype(int)
                gt_binary = ground_truth.astype(int)
                
                tp = np.sum((pred_binary == 1) & (gt_binary == 1))
                fp = np.sum((pred_binary == 1) & (gt_binary == 0))
                fn = np.sum((pred_binary == 0) & (gt_binary == 1))
            else:
                # Simple case for lists
                tp = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt and gt == 1)
                fp = sum(1 for p, gt in zip(predictions, ground_truth) if p == 1 and gt == 0)
                fn = sum(1 for p, gt in zip(predictions, ground_truth) if p == 0 and gt == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Record metrics
            snapshot_precision = MetricsSnapshot(
                timestamp=time.time(),
                metric_name="precision",
                metric_value=precision,
                model_id=model_id,
                additional_data={"tp": tp, "fp": fp, "fn": fn}
            )
            
            snapshot_recall = MetricsSnapshot(
                timestamp=time.time(),
                metric_name="recall",
                metric_value=recall,
                model_id=model_id,
                additional_data={"tp": tp, "fp": fp, "fn": fn}
            )
            
            with self._lock:
                self.metrics_history[f"{model_id}_precision"].append(snapshot_precision)
                self.metrics_history[f"{model_id}_recall"].append(snapshot_recall)
                
                if model_id not in self.model_stats:
                    self.model_stats[model_id] = {}
                self.model_stats[model_id]['precision'] = precision
                self.model_stats[model_id]['recall'] = recall
            
            return precision, recall
            
        except Exception as e:
            logger.error(f"Error calculating precision/recall: {e}")
            return 0.0, 0.0
    
    def calculate_accuracy(
        self,
        predictions: Union[np.ndarray, List],
        ground_truth: Union[np.ndarray, List],
        model_id: str = "unknown"
    ) -> float:
        """
        Calculate accuracy for classification models.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            model_id: Unique identifier for the model
            
        Returns:
            Accuracy value
        """
        try:
            if isinstance(predictions, np.ndarray) and isinstance(ground_truth, np.ndarray):
                correct = np.sum(predictions == ground_truth)
                accuracy = correct / len(ground_truth)
            else:
                correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
                accuracy = correct / len(ground_truth)
            
            # Record accuracy metric
            snapshot = MetricsSnapshot(
                timestamp=time.time(),
                metric_name="accuracy",
                metric_value=accuracy,
                model_id=model_id,
                additional_data={
                    "correct_predictions": correct,
                    "total_predictions": len(ground_truth)
                }
            )
            
            with self._lock:
                self.metrics_history[f"{model_id}_accuracy"].append(snapshot)
                
                if model_id not in self.model_stats:
                    self.model_stats[model_id] = {}
                self.model_stats[model_id]['accuracy'] = accuracy
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0
    
    def monitor_resources(self, model_id: str) -> Dict[str, float]:
        """
        Monitor system resources during inference.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary containing resource usage metrics
        """
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        
        # Get GPU metrics if available
        gpu_metrics = {}
        if torch.cuda.is_available():
            try:
                gpu_metrics = {
                    'gpu_memory_used_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                    'gpu_utilization': self._get_gpu_utilization(),
                    'gpu_memory_total_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
                }
            except Exception as e:
                logger.warning(f"Could not get GPU metrics: {e}")
        
        resource_metrics = {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'memory_percent': memory.percent,
            **gpu_metrics
        }
        
        # Store resource metrics
        with self._lock:
            self.resource_history[model_id].append({
                'timestamp': time.time(),
                'metrics': resource_metrics.copy()
            })
        
        return resource_metrics
    
    def get_performance_report(self, model_id: str) -> Optional[PerformanceReport]:
        """
        Generate a comprehensive performance report for a model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            PerformanceReport object or None if model not found
        """
        with self._lock:
            if model_id not in self.model_stats:
                logger.warning(f"No performance data found for model {model_id}")
                return None
            
            stats = self.model_stats[model_id]
            
            # Calculate metrics
            fps = stats.get('fps', 0.0)
            total_inferences = stats.get('total_inferences', 0)
            avg_inference_time = stats.get('avg_inference_time', 0.0)
            
            # Get accuracy metrics
            accuracy_metrics = {
                'map': stats.get('map', 0.0),
                'precision': stats.get('precision', 0.0),
                'recall': stats.get('recall', 0.0),
                'accuracy': stats.get('accuracy', 0.0)
            }
            
            # Get resource usage
            latest_resources = self._get_latest_resources(model_id)
            
            report = PerformanceReport(
                model_id=model_id,
                total_inferences=total_inferences,
                avg_inference_time=avg_inference_time,
                fps=fps,
                memory_usage_mb=latest_resources.get('memory_mb', 0.0),
                gpu_utilization=latest_resources.get('gpu_utilization'),
                cpu_utilization=latest_resources.get('cpu_percent', 0.0),
                accuracy_metrics=accuracy_metrics,
                custom_metrics={k: v for k, v in stats.items() 
                              if k not in ['map', 'precision', 'recall', 'accuracy', 'fps', 
                                         'total_inferences', 'avg_inference_time', 'last_inference_time']}
            )
            
            return report
    
    def save_performance_report(self, model_id: str, format: str = "json") -> Optional[str]:
        """
        Save performance report to file.
        
        Args:
            model_id: Unique identifier for the model
            format: Output format ('json', 'csv', 'txt')
            
        Returns:
            Path to saved file or None if saving failed
        """
        report = self.get_performance_report(model_id)
        if not report:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_id}_performance_{timestamp}"
        
        try:
            if format == "json":
                filepath = self.output_dir / f"{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(asdict(report), f, indent=2)
                    
            elif format == "csv":
                filepath = self.output_dir / f"{filename}.csv"
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    writer.writerow(['Model ID', report.model_id])
                    writer.writerow(['Total Inferences', report.total_inferences])
                    writer.writerow(['Avg Inference Time (s)', report.avg_inference_time])
                    writer.writerow(['FPS', report.fps])
                    writer.writerow(['Memory Usage (MB)', report.memory_usage_mb])
                    writer.writerow(['GPU Utilization (%)', report.gpu_utilization or 'N/A'])
                    writer.writerow(['CPU Utilization (%)', report.cpu_utilization])
                    
                    for metric, value in report.accuracy_metrics.items():
                        writer.writerow([f'{metric.upper()}', value])
                    
                    for metric, value in report.custom_metrics.items():
                        writer.writerow([metric, value])
            
            elif format == "txt":
                filepath = self.output_dir / f"{filename}.txt"
                with open(filepath, 'w') as f:
                    f.write(f"Performance Report for {model_id}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("General Metrics:\n")
                    f.write(f"  Total Inferences: {report.total_inferences}\n")
                    f.write(f"  Average Inference Time: {report.avg_inference_time:.4f} seconds\n")
                    f.write(f"  FPS: {report.fps:.2f}\n\n")
                    
                    f.write("Resource Usage:\n")
                    f.write(f"  Memory Usage: {report.memory_usage_mb:.2f} MB\n")
                    f.write(f"  GPU Utilization: {report.gpu_utilization:.2f}% (if available)\n")
                    f.write(f"  CPU Utilization: {report.cpu_utilization:.2f}%\n\n")
                    
                    f.write("Accuracy Metrics:\n")
                    for metric, value in report.accuracy_metrics.items():
                        f.write(f"  {metric.upper()}: {value:.4f}\n")
                    
                    if report.custom_metrics:
                        f.write("\nCustom Metrics:\n")
                        for metric, value in report.custom_metrics.items():
                            f.write(f"  {metric}: {value}\n")
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Performance report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return None
    
    def get_metric_history(self, model_id: str, metric_name: str) -> List[MetricsSnapshot]:
        """
        Get history of a specific metric for a model.
        
        Args:
            model_id: Unique identifier for the model
            metric_name: Name of the metric
            
        Returns:
            List of MetricsSnapshot objects
        """
        key = f"{model_id}_{metric_name}"
        with self._lock:
            return list(self.metrics_history[key])
    
    def reset_model_metrics(self, model_id: str) -> None:
        """
        Reset all metrics for a specific model.
        
        Args:
            model_id: Unique identifier for the model
        """
        with self._lock:
            # Remove model-specific metrics
            keys_to_remove = [key for key in self.metrics_history.keys() if key.startswith(f"{model_id}_")]
            for key in keys_to_remove:
                del self.metrics_history[key]
            
            # Remove model stats
            if model_id in self.model_stats:
                del self.model_stats[model_id]
            
            # Reset counters
            if model_id in self.inference_counters:
                del self.inference_counters[model_id]
            if model_id in self.total_inference_times:
                del self.total_inference_times[model_id]
            
            logger.info(f"Metrics reset for model {model_id}")
    
    def _compute_map(
        self, 
        predictions: List[Any], 
        ground_truth: List[Any], 
        iou_threshold: float
    ) -> float:
        """Compute mean Average Precision."""
        # Simplified mAP calculation
        # In a real implementation, this would involve:
        # 1. Calculating IoU between predictions and ground truth
        # 2. Building precision-recall curves
        # 3. Computing AP for each class
        # 4. Taking the mean across all classes
        
        # For now, return a simple score based on prediction accuracy
        if len(predictions) != len(ground_truth):
            return 0.0
        
        # Simple IoU-like metric for demonstration
        correct_predictions = 0
        for pred, gt in zip(predictions, ground_truth):
            if hasattr(pred, 'iou') and hasattr(gt, 'iou'):
                iou = pred.iou(gt) if hasattr(pred, 'iou') else 0.0
                if iou > iou_threshold:
                    correct_predictions += 1
            else:
                # Fallback for non-object detection cases
                if pred == gt:
                    correct_predictions += 1
        
        return correct_predictions / len(ground_truth) if ground_truth else 0.0
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
            
        except Exception:
            return None
    
    def _get_latest_resources(self, model_id: str) -> Dict[str, float]:
        """Get the latest resource metrics for a model."""
        if model_id in self.resource_history and self.resource_history[model_id]:
            return self.resource_history[model_id][-1]['metrics']
        return {}
    
    def _save_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Save a metrics snapshot to file."""
        try:
            filepath = self.output_dir / "metrics_history.json"
            with self._lock:
                # Load existing history
                history_data = {}
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        history_data = json.load(f)
                
                # Add new snapshot
                key = f"{snapshot.model_id}_{snapshot.metric_name}"
                if key not in history_data:
                    history_data[key] = []
                
                history_data[key].append(asdict(snapshot))
                
                # Save updated history
                with open(filepath, 'w') as f:
                    json.dump(history_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error saving metrics snapshot: {e}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()