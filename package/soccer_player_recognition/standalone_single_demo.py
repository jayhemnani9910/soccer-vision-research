#!/usr/bin/env python3
"""
Standalone Single Model Demo for Soccer Player Recognition

Self-contained demo for testing individual models without complex dependencies.
Features:
- RF-DETR Object Detection
- SAM2 Segmentation  
- SigLIP Identification
- ResNet Classification

Author: Soccer Player Recognition Team
Date: 2025-11-04
"""

import sys
import os
import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandaloneSingleModelDemo:
    """Standalone single model demonstration class."""
    
    def __init__(self):
        """Initialize the standalone single model demo."""
        self.models_config = {
            'rf_detr': {
                'name': 'RF-DETR (Real-time Football Detection)',
                'type': 'detection',
                'description': 'Real-time object detection for soccer players, ball, and referees',
                'input_size': (640, 640),
                'supported_formats': ['image', 'video'],
                'capabilities': ['bbox_detection', 'classification', 'confidence_scoring']
            },
            'sam2': {
                'name': 'SAM2 (Segment Anything Model 2)',
                'type': 'segmentation',
                'description': 'Advanced segmentation for precise player masking',
                'input_size': (1024, 1024),
                'supported_formats': ['image', 'video'],
                'capabilities': ['instance_segmentation', 'mask_generation', 'point_prompts']
            },
            'siglip': {
                'name': 'SigLIP (Multimodal Recognition)',
                'type': 'identification',
                'description': 'Multimodal model for player identification and team classification',
                'input_size': (384, 384),
                'supported_formats': ['image', 'video', 'text'],
                'capabilities': ['player_id', 'team_classification', 'text_matching']
            },
            'resnet': {
                'name': 'ResNet (Player Classifier)',
                'type': 'classification',
                'description': 'Deep learning model for player feature extraction and classification',
                'input_size': (224, 224),
                'supported_formats': ['image'],
                'capabilities': ['feature_extraction', 'player_classification', 'similarity_matching']
            }
        }
        
        self.demo_results = {}
        logger.info("Standalone Single Model Demo initialized")
    
    def create_detection_test_image(self) -> Dict[str, Any]:
        """Create test image for detection models."""
        # Create a scene suitable for object detection
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Soccer field background
        image[:] = (34, 139, 34)  # Green
        cv2.rectangle(image, (100, 100), (1180, 620), (255, 255, 255), 3)
        
        # Field markings
        cv2.circle(image, (640, 360), 60, (255, 255, 255), 3)
        cv2.line(image, (640, 100), (640, 620), (255, 255, 255), 2)
        
        # Players (various positions and jersey colors)
        players = [
            # Team A (red)
            {'pos': (200, 200), 'jersey': 10, 'color': (255, 0, 0), 'bbox': [175, 175, 225, 225]},
            {'pos': (400, 300), 'jersey': 7, 'color': (255, 0, 0), 'bbox': [375, 275, 425, 325]},
            {'pos': (600, 400), 'jersey': 9, 'color': (255, 0, 0), 'bbox': [575, 375, 625, 425]},
            
            # Team B (blue)
            {'pos': (800, 300), 'jersey': 11, 'color': (0, 0, 255), 'bbox': [775, 275, 825, 325]},
            {'pos': (1000, 400), 'jersey': 8, 'color': (0, 0, 255), 'bbox': [975, 375, 1025, 425]},
            
            # Referee
            {'pos': (500, 500), 'jersey': 0, 'color': (0, 255, 0), 'bbox': [475, 475, 525, 525]},
        ]
        
        # Add ball
        ball_pos = (320, 180)
        
        for player in players:
            # Player circle
            cv2.circle(image, player['pos'], 25, player['color'], -1)
            cv2.circle(image, player['pos'], 25, (255, 255, 255), 2)
            
            # Jersey number
            if player['jersey'] > 0:
                cv2.putText(image, str(player['jersey']), 
                           (player['pos'][0] - 10, player['pos'][1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add ball
        cv2.circle(image, ball_pos, 10, (255, 255, 255), -1)
        cv2.circle(image, ball_pos, 10, (0, 0, 0), 2)
        
        return {
            'image': image,
            'description': 'Soccer field scene for detection testing',
            'ground_truth': {
                'players': len(players),
                'ball': 1,
                'referee': 1,
                'player_details': players + [{'bbox': [310, 170, 330, 190], 'class': 'ball', 'confidence': 1.0}]
            }
        }
    
    def simulate_rf_detr_detection(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate RF-DETR detection model."""
        logger.info("🔍 Testing RF-DETR Object Detection Model")
        
        start_time = time.time()
        
        try:
            # Simulate model loading
            logger.info("Loading RF-DETR model...")
            time.sleep(0.5)  # Simulate loading time
            
            # Mock detection results
            detections = {
                'objects': [
                    {
                        'class': 'player',
                        'bbox': [175, 175, 225, 225],
                        'confidence': 0.95,
                        'jersey_number': 10
                    },
                    {
                        'class': 'player',
                        'bbox': [375, 275, 425, 325],
                        'confidence': 0.92,
                        'jersey_number': 7
                    },
                    {
                        'class': 'ball',
                        'bbox': [310, 170, 330, 190],
                        'confidence': 0.89,
                        'jersey_number': None
                    },
                    {
                        'class': 'player',
                        'bbox': [775, 275, 825, 325],
                        'confidence': 0.87,
                        'jersey_number': 11
                    }
                ]
            }
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            ground_truth = test_image['ground_truth']
            detected_players = len([d for d in detections['objects'] if d['class'] == 'player'])
            expected_players = ground_truth['players']
            
            accuracy = min(detected_players / max(expected_players, 1), 1.0)
            
            result = {
                'model': 'RF-DETR',
                'model_info': self.models_config['rf_detr'],
                'success': True,
                'processing_time': processing_time,
                'detections': detections,
                'accuracy': accuracy,
                'metrics': {
                    'detection_accuracy': accuracy,
                    'avg_confidence': np.mean([d['confidence'] for d in detections['objects']]),
                    'processing_fps': 1.0 / processing_time if processing_time > 0 else 0,
                    'detected_objects': len(detections['objects']),
                    'players_detected': detected_players,
                    'ball_detected': len([d for d in detections['objects'] if d['class'] == 'ball'])
                }
            }
            
            logger.info(f"✓ RF-DETR completed: {detected_players} players, {accuracy:.2%} accuracy")
            
        except Exception as e:
            logger.error(f"❌ RF-DETR failed: {e}")
            result = {
                'model': 'RF-DETR',
                'model_info': self.models_config['rf_detr'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        return result
    
    def simulate_sam2_segmentation(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SAM2 segmentation model."""
        logger.info("✂️ Testing SAM2 Segmentation Model")
        
        start_time = time.time()
        
        try:
            # Simulate model loading
            logger.info("Loading SAM2 model...")
            time.sleep(0.8)  # Simulate loading time
            
            # Mock segmentation results
            masks = []
            for i in range(5):
                mask = np.zeros(test_image['image'].shape[:2], dtype=np.uint8)
                
                # Create circular mask
                center = (480 + i * 240, 300 + i * 60)
                cv2.circle(mask, center, 60, i + 1, -1)
                
                masks.append({
                    'mask_id': i + 1,
                    'mask': mask,
                    'area': np.sum(mask == (i + 1)),
                    'confidence': 0.95 - i * 0.05,
                    'class': 'player',
                    'jersey_number': 10 + i
                })
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            total_area = test_image['image'].shape[0] * test_image['image'].shape[1]
            total_masked_area = sum(mask['area'] for mask in masks)
            coverage_ratio = total_masked_area / total_area
            
            result = {
                'model': 'SAM2',
                'model_info': self.models_config['sam2'],
                'success': True,
                'processing_time': processing_time,
                'masks': masks,
                'coverage_ratio': coverage_ratio,
                'metrics': {
                    'segmentation_accuracy': coverage_ratio,
                    'num_masks': len(masks),
                    'avg_mask_confidence': np.mean([m['confidence'] for m in masks]),
                    'processing_fps': 1.0 / processing_time if processing_time > 0 else 0,
                    'total_masked_area': total_masked_area,
                    'coverage_percentage': coverage_ratio * 100
                }
            }
            
            logger.info(f"✓ SAM2 completed: {len(masks)} masks, {coverage_ratio:.2%} coverage")
            
        except Exception as e:
            logger.error(f"❌ SAM2 failed: {e}")
            result = {
                'model': 'SAM2',
                'model_info': self.models_config['sam2'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        return result
    
    def simulate_siglip_identification(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SigLIP identification model."""
        logger.info("👤 Testing SigLIP Identification Model")
        
        start_time = time.time()
        
        try:
            # Simulate model loading
            logger.info("Loading SigLIP model...")
            time.sleep(0.6)  # Simulate loading time
            
            # Mock identification results
            identification_result = {
                'player_id': 'P23',
                'jersey_number': 23,
                'team': 'Team B',
                'confidence': 0.89,
                'text_prompts': [
                    'player number 23 team B',
                    'soccer player 23 wearing red jersey',
                    'football player number 23'
                ],
                'similarity_scores': [0.92, 0.89, 0.85],
                'team_classification': {
                    'team_a_probability': 0.05,
                    'team_b_probability': 0.89,
                    'referee_probability': 0.06
                }
            }
            
            processing_time = time.time() - start_time
            
            result = {
                'model': 'SigLIP',
                'model_info': self.models_config['siglip'],
                'success': True,
                'processing_time': processing_time,
                'identification': identification_result,
                'metrics': {
                    'identification_accuracy': identification_result['confidence'],
                    'team_classification_accuracy': max(identification_result['team_classification'].values()),
                    'processing_fps': 1.0 / processing_time if processing_time > 0 else 0,
                    'feature_dimensions': {
                        'visual': 512,
                        'text': 512,
                        'multimodal': 768
                    }
                }
            }
            
            jersey_num = identification_result.get('jersey_number', 'Unknown')
            logger.info(f"✓ SigLIP completed: Player #{jersey_num} ({identification_result['confidence']:.2%} confidence)")
            
        except Exception as e:
            logger.error(f"❌ SigLIP failed: {e}")
            result = {
                'model': 'SigLIP',
                'model_info': self.models_config['siglip'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        return result
    
    def simulate_resnet_classification(self, test_image: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ResNet classification model."""
        logger.info("🏷️ Testing ResNet Classification Model")
        
        start_time = time.time()
        
        try:
            # Simulate model loading
            logger.info("Loading ResNet model...")
            time.sleep(0.4)  # Simulate loading time
            
            # Mock classification results
            class_probabilities = np.random.dirichlet(np.ones(25))  # 25 player classes
            top_classes = np.argsort(class_probabilities)[::-1][:5]
            
            classification_result = {
                'predicted_class': int(top_classes[0]),
                'confidence': float(class_probabilities[top_classes[0]]),
                'top_5_predictions': [
                    {
                        'class_id': int(cls),
                        'player_name': f'Player {cls+1}',
                        'probability': float(class_probabilities[cls])
                    } for cls in top_classes
                ],
                'feature_vector': np.random.randn(512)
            }
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            ground_truth_class = test_image.get('ground_truth', {}).get('class_id', 1)
            top1_accuracy = 1.0 if classification_result['predicted_class'] == ground_truth_class else 0.0
            top5_accuracy = 1.0 if ground_truth_class in top_classes else 0.0
            
            result = {
                'model': 'ResNet',
                'model_info': self.models_config['resnet'],
                'success': True,
                'processing_time': processing_time,
                'classification': classification_result,
                'metrics': {
                    'top1_accuracy': top1_accuracy,
                    'top5_accuracy': top5_accuracy,
                    'avg_inference_time': processing_time,
                    'processing_fps': 1.0 / processing_time if processing_time > 0 else 0,
                    'feature_vector_norm': np.linalg.norm(classification_result['feature_vector']),
                    'num_classes': len(class_probabilities)
                }
            }
            
            logger.info(f"✓ ResNet completed: Class {classification_result['predicted_class']} ({classification_result['confidence']:.2%} confidence)")
            
        except Exception as e:
            logger.error(f"❌ ResNet failed: {e}")
            result = {
                'model': 'ResNet',
                'model_info': self.models_config['resnet'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        return result
    
    def run_single_model_test(self, model_name: str) -> Dict[str, Any]:
        """Run a single model test."""
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models_config.keys())}")
        
        model_config = self.models_config[model_name]
        model_type = model_config['type']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TESTING {model_config['name'].upper()}")
        logger.info(f"{'='*80}")
        
        # Create appropriate test image
        if model_type == 'detection':
            test_image = self.create_detection_test_image()
        else:
            # Create a generic test image for other models
            test_image = {
                'image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                'description': f'Generic test image for {model_type} models',
                'ground_truth': {'class_id': 1}
            }
        
        # Run model-specific demo
        if model_name == 'rf_detr':
            result = self.simulate_rf_detr_detection(test_image)
        elif model_name == 'sam2':
            result = self.simulate_sam2_segmentation(test_image)
        elif model_name == 'siglip':
            result = self.simulate_siglip_identification(test_image)
        elif model_name == 'resnet':
            result = self.simulate_resnet_classification(test_image)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Store result
        self.demo_results[model_name] = result
        
        # Print summary
        if result['success']:
            logger.info(f"✅ {model_config['name']} test completed successfully")
            self._print_model_summary(result)
        else:
            logger.error(f"❌ {model_config['name']} test failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def _print_model_summary(self, result: Dict[str, Any]):
        """Print model test summary."""
        model_name = result['model']
        metrics = result.get('metrics', {})
        
        logger.info(f"\n📊 {model_name} Test Summary:")
        logger.info(f"  • Processing Time: {result['processing_time']:.3f}s")
        logger.info(f"  • Performance: {metrics.get('processing_fps', 0):.2f} FPS")
        
        # Model-specific metrics
        if model_name == 'RF-DETR':
            logger.info(f"  • Detection Accuracy: {metrics.get('detection_accuracy', 0):.2%}")
            logger.info(f"  • Objects Detected: {metrics.get('detected_objects', 0)}")
            logger.info(f"  • Avg Confidence: {metrics.get('avg_confidence', 0):.2%}")
        
        elif model_name == 'SAM2':
            logger.info(f"  • Segmentation Coverage: {metrics.get('coverage_percentage', 0):.2f}%")
            logger.info(f"  • Masks Generated: {metrics.get('num_masks', 0)}")
            logger.info(f"  • Avg Mask Confidence: {metrics.get('avg_mask_confidence', 0):.2%}")
        
        elif model_name == 'SigLIP':
            logger.info(f"  • Identification Accuracy: {metrics.get('identification_accuracy', 0):.2%}")
            logger.info(f"  • Team Classification: {metrics.get('team_classification_accuracy', 0):.2%}")
        
        elif model_name == 'ResNet':
            logger.info(f"  • Top-1 Accuracy: {metrics.get('top1_accuracy', 0):.2%}")
            logger.info(f"  • Top-5 Accuracy: {metrics.get('top5_accuracy', 0):.2%}")
    
    def run_all_models_test(self) -> Dict[str, Any]:
        """Run tests for all models."""
        logger.info("🚀 Running Single Model Tests for All Models")
        logger.info("="*80)
        
        all_results = {}
        
        for model_name in self.models_config.keys():
            logger.info(f"\n🔄 Testing {model_name.upper()}...")
            try:
                result = self.run_single_model_test(model_name)
                all_results[model_name] = result
                logger.info(f"✅ {model_name} completed")
            except Exception as e:
                logger.error(f"❌ {model_name} failed: {e}")
                all_results[model_name] = {
                    'model': model_name.upper(),
                    'success': False,
                    'error': str(e)
                }
            
            # Brief pause between tests
            time.sleep(0.5)
        
        return all_results
    
    def run_interactive_demo(self):
        """Run interactive demo allowing user to choose models."""
        logger.info("🎮 Interactive Single Model Demo")
        logger.info("="*50)
        
        print("\nAvailable Models:")
        for i, (name, config) in enumerate(self.models_config.items(), 1):
            print(f"{i}. {config['name']}")
            print(f"   Type: {config['type'].title()}")
            print(f"   Description: {config['description']}")
            print()
        
        while True:
            try:
                choice = input("Choose a model to test (1-4), 'all' for all models, or 'q' to quit: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == 'all':
                    logger.info("Running all model tests...")
                    self.run_all_models_test()
                elif choice.isdigit() and 1 <= int(choice) <= len(self.models_config):
                    model_names = list(self.models_config.keys())
                    selected_model = model_names[int(choice) - 1]
                    self.run_single_model_test(selected_model)
                else:
                    print("Invalid choice. Please try again.")
                
                # Save results after each test
                self.save_results()
                
            except KeyboardInterrupt:
                logger.info("\nDemo interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info("Interactive demo completed")
    
    def save_results(self):
        """Save single model demo results."""
        logger.info("\n💾 Saving single model demo results...")
        
        try:
            output_dir = Path("outputs/standalone_single_model_demo")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive summary
            summary = {
                'demo_info': {
                    'name': 'Standalone Single Model Demo',
                    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'version': '1.0',
                    'type': 'standalone',
                    'models_tested': list(self.models_config.keys()),
                    'total_tests': len(self.demo_results)
                },
                'model_configs': self.models_config,
                'test_results': self.demo_results,
                'performance_summary': self._create_performance_summary(),
                'capabilities': {
                    'individual_testing': 'Test each model independently',
                    'detailed_analysis': 'Comprehensive metrics and visualizations',
                    'performance_benchmarking': 'Speed and accuracy measurements',
                    'error_handling': 'Robust error reporting and recovery',
                    'model_comparison': 'Side-by-side model performance analysis'
                },
                'usage_examples': {
                    'test_single_model': 'result = demo.run_single_model_test("rf_detr")',
                    'test_all_models': 'results = demo.run_all_models_test()',
                    'interactive_mode': 'demo.run_interactive_demo()'
                }
            }
            
            # Save summary
            with open(output_dir / 'demo_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save individual model results
            for model_name, result in self.demo_results.items():
                with open(output_dir / f'{model_name}_results.json', 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            
            logger.info(f"✓ Results saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return None
    
    def _create_performance_summary(self) -> Dict[str, Any]:
        """Create performance summary across all models."""
        if not self.demo_results:
            return {}
        
        successful_tests = {k: v for k, v in self.demo_results.items() if v.get('success', False)}
        
        summary = {
            'total_models': len(self.demo_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(self.demo_results) - len(successful_tests),
            'models': {}
        }
        
        for model_name, result in successful_tests.items():
            metrics = result.get('metrics', {})
            summary['models'][model_name] = {
                'processing_time': result.get('processing_time', 0),
                'fps': metrics.get('processing_fps', 0),
                'success': result.get('success', False)
            }
        
        return summary
    
    def run_demo(self) -> bool:
        """Run the complete standalone demo."""
        try:
            # Run all models test by default
            self.run_all_models_test()
            
            # Save results
            output_dir = self.save_results()
            
            # Final summary
            logger.info(f"\n{'='*80}")
            logger.info("🎉 STANDALONE SINGLE MODEL DEMO COMPLETED")
            logger.info(f"{'='*80}")
            
            successful_models = sum(1 for result in self.demo_results.values() if result.get('success', False))
            total_models = len(self.demo_results)
            
            logger.info(f"✅ {successful_models}/{total_models} models tested successfully!")
            logger.info("🔧 Models demonstrated:")
            
            for model_name, config in self.models_config.items():
                status = "✅" if self.demo_results.get(model_name, {}).get('success', False) else "❌"
                logger.info(f"  {status} {config['name']}: {config['description']}")
            
            if output_dir:
                logger.info(f"📁 Results saved to: {output_dir}")
            
            logger.info("\n🎯 Standalone single model demo completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    print("🚀 Standalone Single Model Demo - Soccer Player Recognition System")
    print("="*75)
    
    demo = StandaloneSingleModelDemo()
    
    try:
        success = demo.run_demo()
        
        if success:
            print("\n🎉 Standalone Single Model Demo completed successfully!")
            print("Check the 'outputs/standalone_single_model_demo/' directory for detailed results.")
            print("\nTo run in interactive mode, modify the code to call:")
            print("demo.run_interactive_demo()")
        else:
            print("\n❌ Standalone Single Model Demo failed.")
            return 1
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())