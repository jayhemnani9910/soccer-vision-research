"""
Test package for Soccer Player Recognition System.

This package contains comprehensive testing frameworks including:
- Model testing (RF-DETR, SAM2, SigLIP, ResNet, MediaPipe)
- Performance benchmarking and regression testing
- Integration testing for end-to-end workflows
- Utility testing for all supporting modules
"""

# Make test utilities available
from .test_all_models import (
    TestModelInstances,
    TestModelManager,
    TestModelTypes,
    TestModelConfiguration,
    create_test_suite as create_model_test_suite
)

from .performance_tests import (
    TestModelLoadingPerformance,
    TestInferencePerformance,
    TestBatchProcessingPerformance,
    TestDevicePerformance,
    TestScalabilityPerformance,
    TestPerformanceRegression,
    create_performance_test_suite
)

from .integration_tests import (
    TestPipelineIntegration,
    TestDataFlowIntegration,
    TestConfigurationIntegration,
    TestErrorHandlingIntegration,
    TestPerformanceIntegration,
    create_integration_test_suite
)

from .utils_tests import (
    TestConfigLoader,
    TestPerformanceMonitor,
    TestImageUtils,
    TestVideoUtils,
    TestDrawUtils,
    TestLogger,
    TestModelSpecificUtils,
    create_utility_test_suite
)

# Version information
__version__ = "1.0.0"
__author__ = "Soccer Player Recognition Team"

# Test categories
TEST_CATEGORIES = {
    "models": "Model functionality and interface testing",
    "performance": "Performance benchmarking and regression testing",
    "integration": "End-to-end workflow and component integration",
    "utilities": "Utility module and helper function testing"
}

# Default test configuration
DEFAULT_TEST_CONFIG = {
    "device": "cpu",  # or "cuda" if available
    "batch_size": 1,
    "timeout": 300,  # seconds
    "verbose": True,
    "coverage": False,
    "parallel": False
}

def get_test_config():
    """Get test configuration from environment or defaults."""
    import os
    
    config = DEFAULT_TEST_CONFIG.copy()
    
    # Override with environment variables
    if os.getenv('TEST_DEVICE'):
        config['device'] = os.getenv('TEST_DEVICE')
    
    if os.getenv('TEST_BATCH_SIZE'):
        config['batch_size'] = int(os.getenv('TEST_BATCH_SIZE'))
    
    if os.getenv('TEST_TIMEOUT'):
        config['timeout'] = int(os.getenv('TEST_TIMEOUT'))
    
    if os.getenv('TEST_VERBOSE'):
        config['verbose'] = os.getenv('TEST_VERBOSE').lower() == 'true'
    
    if os.getenv('TEST_COVERAGE'):
        config['coverage'] = os.getenv('TEST_COVERAGE').lower() == 'true'
    
    if os.getenv('TEST_PARALLEL'):
        config['parallel'] = os.getenv('TEST_PARALLEL').lower() == 'true'
    
    return config

def run_all_tests():
    """Run all test suites."""
    import sys
    import time
    import unittest
    
    start_time = time.time()
    
    # Create and run test suites
    test_suites = [
        ("Models", create_model_test_suite()),
        ("Performance", create_performance_test_suite()),
        ("Integration", create_integration_test_suite()),
        ("Utilities", create_utility_test_suite())
    ]
    
    results = {}
    
    for name, suite in test_suites:
        print(f"\n{'='*60}")
        print(f"Running {name} Tests")
        print(f"{'='*60}")
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        results[name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful()
        }
    
    # Print summary
    total_time = time.time() - start_time
    total_tests = sum(r['tests_run'] for r in results.values())
    total_failures = sum(r['failures'] for r in results.values())
    total_errors = sum(r['errors'] for r in results.values())
    
    print(f"\n{'='*60}")
    print(f"Test Summary (Total time: {total_time:.2f}s)")
    print(f"{'='*60}")
    
    for name, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{name:15}: {result['tests_run']:3d} tests, {result['failures']:2d} failures, {result['errors']:2d} errors - {status}")
    
    print(f"{'='*60}")
    print(f"Total: {total_tests} tests, {total_failures} failures, {total_errors} errors")
    
    success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    return all(r['success'] for r in results.values())

def run_specific_tests(test_category: str, test_names: list = None):
    """Run specific test categories or individual tests."""
    import unittest
    import sys
    
    suite_creators = {
        'models': create_model_test_suite,
        'performance': create_performance_test_suite,
        'integration': create_integration_test_suite,
        'utilities': create_utility_test_suite
    }
    
    if test_category not in suite_creators:
        print(f"Unknown test category: {test_category}")
        print(f"Available categories: {list(suite_creators.keys())}")
        return False
    
    if test_names:
        # Run specific test methods
        suite = suite_creators[test_category]()
        test_suite = unittest.TestSuite()
        
        for test_name in test_names:
            for test_case in suite:
                if test_name in str(test_case):
                    test_suite.addTest(test_case)
        
        if len(test_suite) == 0:
            print(f"No tests found matching: {test_names}")
            return False
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        return result.wasSuccessful()
    else:
        # Run entire category
        suite = suite_creators[test_category]()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()

def check_test_dependencies():
    """Check if all test dependencies are available."""
    import sys
    
    dependencies = {
        'torch': 'PyTorch for deep learning models',
        'numpy': 'NumPy for numerical operations',
        'cv2': 'OpenCV for image/video processing',
        'yaml': 'PyYAML for configuration parsing',
        'psutil': 'psutil for system monitoring',
        'unittest': 'Python unittest framework'
    }
    
    missing_deps = []
    available_deps = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            available_deps.append(f"{dep}: {description}")
        except ImportError:
            missing_deps.append(f"{dep}: {description}")
    
    print("Available Dependencies:")
    for dep in available_deps:
        print(f"  ✅ {dep}")
    
    if missing_deps:
        print("\nMissing Dependencies:")
        for dep in missing_deps:
            print(f"  ❌ {dep}")
        print("\nInstall missing dependencies with:")
        print("pip install " + " ".join(dep.split(':')[0] for dep in missing_deps))
        return False
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Soccer Player Recognition Test Runner')
    parser.add_argument('--category', choices=['models', 'performance', 'integration', 'utilities'],
                       help='Run specific test category')
    parser.add_argument('--tests', nargs='*', help='Run specific test methods')
    parser.add_argument('--check-deps', action='store_true', help='Check test dependencies')
    parser.add_argument('--config', action='store_true', help='Show current test configuration')
    
    args = parser.parse_args()
    
    if args.check_deps:
        success = check_test_dependencies()
        sys.exit(0 if success else 1)
    
    if args.config:
        config = get_test_config()
        print("Current Test Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        sys.exit(0)
    
    if args.category:
        success = run_specific_tests(args.category, args.tests)
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)