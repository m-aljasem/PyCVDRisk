#!/usr/bin/env python3
"""
Local build script for CVD Risk Calculator

This script provides convenient commands for building, testing, and publishing
the package locally.

Usage:
    python build.py [command]

Commands:
    build         - Build source distribution and wheels
    test          - Run tests with coverage
    lint          - Run linters (black, ruff, mypy)
    check         - Run all checks (lint + test)
    clean         - Clean build artifacts
    install       - Install package in development mode
    publish-test  - Publish to TestPyPI
    publish       - Publish to PyPI (be careful!)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        sys.exit(1)


def build():
    """Build source distribution and wheels."""
    run_command("python -m build", "Building package")


def test():
    """Run tests with coverage."""
    run_command("pytest --cov=src/cvd_risk_calculator --cov-report=term-missing", "Running tests")


def lint():
    """Run linters."""
    run_command("black --check src tests", "Checking code formatting")
    run_command("ruff check src tests", "Checking code quality")
    run_command("mypy src", "Checking type hints")


def check():
    """Run all checks."""
    lint()
    test()


def clean():
    """Clean build artifacts."""
    print("🧹 Cleaning build artifacts...")
    dirs_to_clean = ["dist", "build", "*.egg-info", ".coverage", "htmlcov", ".pytest_cache"]
    files_to_clean = ["coverage.xml", ".coverage"]

    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed directory: {path}")

    for pattern in files_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_file():
                path.unlink()
                print(f"  Removed file: {path}")

    print("✅ Clean completed")


def install():
    """Install package in development mode."""
    run_command("pip install -e .[dev]", "Installing package in development mode")


def publish_test():
    """Publish to TestPyPI."""
    build()
    run_command("twine check dist/*", "Checking distribution")
    run_command("twine upload --repository testpypi dist/*", "Publishing to TestPyPI")


def publish():
    """Publish to PyPI."""
    if not confirm_action("Are you sure you want to publish to PyPI? This will make the package publicly available."):
        print("❌ Publish cancelled")
        return

    build()
    run_command("twine check dist/*", "Checking distribution")
    run_command("twine upload dist/*", "Publishing to PyPI")


def confirm_action(message):
    """Ask user to confirm an action."""
    response = input(f"{message} (y/N): ").strip().lower()
    return response in ['y', 'yes']


def main():
    parser = argparse.ArgumentParser(description="Local build script for CVD Risk Calculator")
    parser.add_argument(
        "command",
        choices=["build", "test", "lint", "check", "clean", "install", "publish-test", "publish"],
        help="Command to run"
    )

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run from the project root.")
        sys.exit(1)

    # Run the requested command
    command_func = globals()[args.command.replace("-", "_")]
    command_func()


if __name__ == "__main__":
    main()
