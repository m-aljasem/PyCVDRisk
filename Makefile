.PHONY: help install install-dev test test-cov lint format type-check clean docs

help:
	@echo "Available commands:"
	@echo "  make install       - Install package in production mode"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run linters (ruff, black check)"
	@echo "  make format        - Format code (black, ruff)"
	@echo "  make type-check    - Run type checker (mypy)"
	@echo "  make docs          - Build Sphinx documentation"
	@echo "  make clean         - Clean build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,performance]"

test:
	pytest

test-cov:
	pytest --cov=src/cvd_risk --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

type-check:
	mypy src

docs:
	cd docs && make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

