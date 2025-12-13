# Makefile for Soccer Player Recognition Testing Framework
# This file provides convenient commands for testing, linting, and development tasks

.PHONY: help install install-dev test test-all test-unit test-integration test-performance test-utils clean lint format type-check security docs build

# Default target
help:  ## Show this help message
	@echo "Soccer Player Recognition - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance tests only"
	@echo "  test-models       Run model tests only"
	@echo "  test-utils        Run utility tests only"
	@echo "  test-fast         Run fast tests only (skip slow tests)"
	@echo "  test-gpu          Run GPU tests only"
	@echo "  test-parallel     Run tests in parallel"
	@echo "  test-coverage     Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint              Run code linting"
	@echo "  format            Format code with black and isort"
	@echo "  type-check        Run type checking with mypy"
	@echo "  security          Run security checks"
	@echo "  check-all         Run all code quality checks"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install           Install project dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  install-test      Install testing dependencies"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean             Clean up build artifacts and caches"
	@echo "  docs              Build documentation"
	@echo "  build             Build the project"
	@echo "  deps-check        Check dependencies"
	@echo ""
	@echo "CI Commands:"
	@echo "  ci                Run CI pipeline locally"
	@echo "  test-ci           Run tests as in CI"

# Installation targets
install:  ## Install project dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e .

install-test:  ## Install testing dependencies only
	pip install -r requirements-test.txt

# Testing targets
test:  ## Run all tests
	python -m pytest tests/ -v

test-unit:  ## Run unit tests only
	python -m pytest tests/test_all_models.py tests/utils_tests.py -v

test-integration:  ## Run integration tests only
	python -m pytest tests/integration_tests.py -v

test-performance:  ## Run performance tests only
	python -m pytest tests/performance_tests.py -v -m "not slow"

test-models:  ## Run model tests only
	python -m pytest tests/test_all_models.py -v

test-utils:  ## Run utility tests only
	python -m pytest tests/utils_tests.py -v

test-fast:  ## Run fast tests only (skip slow tests)
	python -m pytest tests/ -v -m "not slow"

test-gpu:  ## Run GPU tests only
	python -m pytest tests/ -v -m "gpu"

test-parallel:  ## Run tests in parallel (requires pytest-xdist)
	python -m pytest tests/ -n auto -v

test-coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing -v

test-quick:  ## Run quick smoke tests
	python -m pytest tests/utils_tests.py::TestConfigLoader::test_config_loader_initialization -v
	python -m pytest tests/test_all_models.py::TestModelInstances::test_model_instance_creation -v

# CI simulation
test-ci:  ## Run tests as in CI (subset for local testing)
	python -m pytest tests/test_all_models.py -v --tb=short
	python -m pytest tests/utils_tests.py -v --tb=short
	python -m pytest tests/integration_tests.py -v --tb=short -k "not slow"

# Individual test scripts
test-models-script:  ## Run model tests using test script
	python tests/test_all_models.py

test-performance-script:  ## Run performance tests using test script
	python tests/performance_tests.py

test-integration-script:  ## Run integration tests using test script
	python tests/integration_tests.py

test-utils-script:  ## Run utility tests using test script
	python tests/utils_tests.py

# Code quality targets
lint:  ## Run code linting with flake8
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:  ## Format code with black and isort
	black .
	isort .

format-check:  ## Check code formatting
	black --check --diff .
	isort --check-only --diff .

type-check:  ## Run type checking with mypy
	mypy . --ignore-missing-imports

security:  ## Run security checks
	bandit -r . -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

check-all: lint type-check security  ## Run all code quality checks

# Documentation targets
docs:  ## Build documentation
	cd docs && sphinx-build -W -b html . _build/html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Build targets
build:  ## Build the project
	python setup.py build

clean:  ## Clean up build artifacts and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f bandit-report.json
	rm -f safety-report.json
	find . -type f -name "*.log" -delete

# Dependency management
deps-check:  ## Check dependencies for issues
	pip check
	pip list --outdated

deps-update:  ## Update dependencies to latest versions
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

# CI simulation
ci:  ## Run CI pipeline locally (subset of GitHub Actions)
	@echo "Running CI pipeline locally..."
	make check-all
	make test-ci
	@echo "CI pipeline completed"

# Debug and development utilities
test-debug:  ## Run tests with debugging output
	python -m pytest tests/ -v -s --tb=long --capture=no

test-failed:  ## Run only previously failed tests
	python -m pytest tests/ --lf

test-last-failed:  ## Run tests and stop on first failure
	python -m pytest tests/ -x

profile:  ## Run performance profiling on tests
	python -m pytest tests/ --profile

coverage-report:  ## Generate detailed coverage report
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Environment management
env-setup:  ## Set up development environment
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && make install-dev

env-clean:  ## Clean virtual environment
	rm -rf venv/

# Test data management
test-data-clean:  ## Clean test data
	rm -rf /tmp/soccer_player_recognition_tests_*
	find . -name "test_*.jpg" -delete
	find . -name "test_*.mp4" -delete
	find . -name "test_*.pt" -delete

# Performance benchmarks
benchmark:  ## Run performance benchmarks
	python tests/performance_tests.py

benchmark-compare:  ## Compare performance against baseline
	python -c "
	import json
	import os
	baseline_file = 'baseline_performance.json'
	if os.path.exists(baseline_file):
		print('Baseline performance metrics:')
		with open(baseline_file) as f:
			data = json.load(f)
		for metric in data:
			print(f'{metric[\"operation\"]}: {metric[\"execution_time\"]:.4f}s')
	else:
		print('No baseline found. Run tests first to create baseline.')
	"

# Version and info
version:  ## Show version information
	@python -c "import sys; print(f'Python: {sys.version}')"
	@pip list | grep -E "(torch|numpy|opencv|pytest)"

info:  ## Show project information
	@echo "Project: Soccer Player Recognition System"
	@echo "Testing Framework Version: 1.0.0"
	@make version