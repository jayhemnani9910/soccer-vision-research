#!/usr/bin/env python3
"""
Test Runner CLI for Soccer Player Recognition System

This script provides a convenient command-line interface for running different types of tests
with various configurations and options.
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path

def run_command(cmd, capture_output=True, timeout=300, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"


def install_dependencies():
    """Install test dependencies."""
    print("Installing test dependencies...")
    
    # Get current directory
    current_dir = Path.cwd()
    requirements_file = current_dir / "requirements-test.txt"
    if not requirements_file.exists():
        requirements_file = current_dir / "requirements-dev.txt"
    
    if requirements_file.exists():
        success, stdout, stderr = run_command(f"pip install -r {requirements_file}", cwd=current_dir)
        if success:
            print("✅ Dependencies installed successfully")
        else:
            print(f"❌ Failed to install dependencies: {stderr}")
            return False
    else:
        print("❌ Requirements file not found")
        return False
    
    return True


def setup_test_environment():
    """Set up the test environment."""
    print("Setting up test environment...")
    
    # Create test directories
    current_dir = Path.cwd()
    test_dirs = [
        "tmp",
        "outputs/test_results",
        "test_data"
    ]
    
    for dir_path in test_dirs:
        full_path = current_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Test environment setup complete")


def run_tests_with_pytest(test_type=None, test_names=None, **kwargs):
    """Run tests using pytest."""
    current_dir = Path.cwd()
    cmd_parts = ["python", "-m", "pytest"]
    
    # Add test path
    test_path = "tests"
    if test_type:
        test_path = f"tests/{test_type}*.py"
    
    cmd_parts.append(test_path)
    
    # Add pytest options
    options = []
    
    if kwargs.get('verbose', False):
        options.append("-v")
    
    if kwargs.get('coverage', False):
        options.extend(["--cov=. ", "--cov-report=html", "--cov-report=term"])
    
    if kwargs.get('parallel', False):
        options.append("-n")
        options.append("auto")
    
    if kwargs.get('timeout'):
        options.append(f"--timeout={kwargs['timeout']}")
    
    if kwargs.get('markers'):
        options.extend(["-m", kwargs['markers']])
    
    if kwargs.get('output_format'):
        if kwargs['output_format'] == 'html':
            options.append("--html=test_report.html")
    
    if kwargs.get('fast', False):
        options.append("-x")  # Stop on first failure
        options.append("--tb=short")  # Shorter traceback
    
    # Add specific test names if provided
    if test_names:
        for test_name in test_names:
            options.append(f"-k {test_name}")
    
    cmd_parts.extend(options)
    cmd = " ".join(cmd_parts)
    
    print(f"Running command: {cmd}")
    success, stdout, stderr = run_command(cmd, capture_output=False, cwd=current_dir)
    
    if success:
        print("✅ Tests completed successfully")
    else:
        print("❌ Tests failed")
        if stderr:
            print(f"Error output: {stderr}")
    
    return success


def run_benchmark_comparison():
    """Run performance benchmark comparison."""
    print("Running performance benchmark comparison...")
    
    current_dir = Path.cwd()
    baseline_file = current_dir / "tmp" / "baseline_performance.json"
    
    if baseline_file.exists():
        print("📊 Comparing against existing baseline...")
        success = run_command("python -m pytest tests/performance_tests.py::TestPerformanceRegression -v", cwd=current_dir)
    else:
        print("📊 No baseline found. Creating new baseline...")
        success = run_command("python -m pytest tests/performance_tests.py::TestPerformanceRegression::test_baseline_performance -v", cwd=current_dir)
    
    return success


def check_test_dependencies():
    """Check if test dependencies are available."""
    required_packages = [
        'pytest',
        'numpy',
        'opencv-python',
        'torch',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'opencv-python':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements-test.txt")
        return False
    else:
        print("✅ All test dependencies are available")
        return True


def show_test_categories():
    """Show available test categories."""
    print("Available test categories:")
    categories = {
        'models': 'Model functionality and interface tests',
        'performance': 'Performance benchmarking and regression tests',
        'integration': 'End-to-end workflow and component integration',
        'utilities': 'Utility module and helper function tests'
    }
    
    for category, description in categories.items():
        print(f"  {category}: {description}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test Runner for Soccer Player Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          # Run all tests
  %(prog)s --category models              # Run model tests only
  %(prog)s --performance --coverage       # Run performance tests with coverage
  %(prog)s --integration --verbose        # Run integration tests with verbose output
  %(prog)s --utils --fast                 # Run utility tests, skip slow tests
  %(prog)s --setup                        # Set up test environment
  %(prog)s --deps-check                   # Check dependencies
  %(prog)s --benchmark-compare            # Compare performance against baseline
  %(prog)s --clean                        # Clean test artifacts
  %(prog)s --list-categories              # Show available test categories
        """
    )
    
    # Main options
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--category', choices=['models', 'performance', 'integration', 'utils'],
                       help='Run specific test category')
    parser.add_argument('--setup', action='store_true', help='Set up test environment and dependencies')
    parser.add_argument('--deps-check', action='store_true', help='Check test dependencies')
    parser.add_argument('--list-categories', action='store_true', help='Show available test categories')
    
    # Test options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--timeout', type=int, default=300, help='Test timeout in seconds')
    parser.add_argument('--markers', help='pytest markers to filter tests')
    parser.add_argument('--output-format', choices=['html', 'json'], help='Output format for reports')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only (skip slow)')
    
    # Specific tests
    parser.add_argument('--test', action='append', help='Run specific test (can be used multiple times)')
    parser.add_argument('--benchmark-compare', action='store_true', help='Compare performance against baseline')
    
    # Development options
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--clean', action='store_true', help='Clean test artifacts')
    
    args = parser.parse_args()
    
    # Handle setup
    if args.setup:
        print("Setting up test environment...")
        if args.install_deps:
            if not install_dependencies():
                return 1
        setup_test_environment()
        print("✅ Setup complete")
        return 0
    
    # Handle dependency check
    if args.deps_check:
        if not check_test_dependencies():
            return 1
        return 0
    
    # Handle list categories
    if args.list_categories:
        show_test_categories()
        return 0
    
    # Handle benchmark comparison
    if args.benchmark_compare:
        return 0 if run_benchmark_comparison() else 1
    
    # Handle cleaning
    if args.clean:
        print("Cleaning test artifacts...")
        import shutil
        current_dir = Path.cwd()
        temp_dirs = list(current_dir.glob("tmp/*"))
        for temp_dir in temp_dirs:
            if temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Cleanup complete")
        return 0
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            return 1
    
    # Handle test execution
    start_time = time.time()
    
    # Convert args to kwargs for pytest
    pytest_kwargs = {
        'verbose': args.verbose,
        'coverage': args.coverage,
        'parallel': args.parallel,
        'timeout': args.timeout,
        'markers': args.markers,
        'output_format': args.output_format,
        'fast': args.fast
    }
    
    if args.all:
        print("Running all tests...")
        success = run_tests_with_pytest(**pytest_kwargs)
    elif args.category:
        print(f"Running {args.category} tests...")
        success = run_tests_with_pytest(test_type=args.category, test_names=args.test, **pytest_kwargs)
    else:
        # Default to running all tests
        print("Running all tests (default)...")
        success = run_tests_with_pytest(**pytest_kwargs)
    
    # Show execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Show test summary
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        print("\nTroubleshooting tips:")
        print("1. Check the error messages above")
        print("2. Ensure all dependencies are installed: --install-deps")
        print("3. Run with --verbose for more detailed output")
        print("4. Check test logs in tmp/ directory")
        print("5. Use --deps-check to verify dependencies")
        return 1


if __name__ == "__main__":
    sys.exit(main())