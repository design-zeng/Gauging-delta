# Makefile for Gauging-δ development
# Usage: make <target>

.PHONY: help install dev test lint format typecheck security clean all check benchmark bench-test bench-full bench-parity bench-parity-deep bench-stress bench-stress-deep gate-parity cycle-profile cycle-bench cycle-verify

# Default target
help:
	@echo "Gauging-δ Development Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  make install    Install package in editable mode"
	@echo "  make dev        Install with dev dependencies"
	@echo "  make test       Run test suite"
	@echo "  make lint       Run linter (ruff)"
	@echo "  make format     Format code (ruff)"
	@echo "  make typecheck  Run type checker (mypy)"
	@echo "  make security   Run security scanner (bandit)"
	@echo "  make check      Run all checks (lint, typecheck, test)"
	@echo "  make benchmark  Run full benchmarks (standalone)"
	@echo "  make bench-test Run benchmark tests (pytest)"
	@echo "  make bench-full Run full benchmark suite (60 GB RAM)"
	@echo "  make bench-parity      Parity benchmark (holistic)"
	@echo "  make bench-parity-deep Parity benchmark (component level)"
	@echo "  make bench-stress      Stress test (100 configs)"
	@echo "  make bench-stress-deep Stress test (deep mode)"
	@echo "  make gate-parity       Gate-by-gate parity verification"
	@echo "  make cycle-profile     Profile hotspots + memory + N=70K estimate"
	@echo "  make cycle-bench       Benchmark vs previous cycle baseline"
	@echo "  make cycle-verify      Targeted gate parity (GATES=2,3)"
	@echo "  make clean      Remove build artifacts"
	@echo "  make all        Install and run all checks"
	@echo ""

# Installation
install:
	uv sync

dev:
	uv sync --extra dev
	pre-commit install

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/gauging_delta --cov-report=term-missing --cov-report=html

# Linting and formatting
lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

format-check:
	ruff format src/ tests/ --check

# Type checking
typecheck:
	mypy src/gauging_delta/

# Security
security:
	bandit -r src/ -c pyproject.toml

# Benchmarks
benchmark:
	uv run python benchmarks/benchmark_all.py

bench-test:
	uv run pytest benchmarks/test_performance.py -v --tb=short

bench-full:
	uv run --extra bench python benchmarks/run_full_benchmark.py

bench-parity:
	uv run python benchmarks/benchmark_parity.py

bench-parity-deep:
	uv run python benchmarks/benchmark_parity.py --level component --sizes 100 500 1000

bench-stress:
	uv run python benchmarks/stress_test.py

bench-stress-deep:
	uv run python benchmarks/stress_test.py --deep

gate-parity:
	uv run python benchmarks/gate_parity.py

cycle-profile:
	uv run python benchmarks/cycle_bench.py profile

cycle-bench:
	uv run python benchmarks/cycle_bench.py bench

cycle-verify:
	uv run python benchmarks/cycle_bench.py verify $(if $(GATES),--gates $(GATES),)

# Combined checks
check: lint typecheck test

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf benchmarks/results/

# Full workflow
all: dev check
	@echo ""
	@echo "✓ All checks passed!"
