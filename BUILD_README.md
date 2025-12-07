# 🏗️ Build System Setup

This document explains the automated build and release system for the CVD Risk Calculator package.

## 📁 Files Overview

### GitHub Actions Workflows
- **`.github/workflows/ci.yml`** - Continuous Integration (tests, linting, coverage)
- **`.github/workflows/release.yml`** - Automated PyPI publishing on releases
- **`.github/workflows/build-wheels.yml`** - Multi-platform wheel building

### Build Scripts
- **`build_script.py`** - Local development build utilities
- **`release.py`** - Version management and release preparation
- **`tox.ini`** - Multi-environment testing configuration

### Configuration Files
- **`pyproject.toml`** - Package metadata and build configuration
- **`.pre-commit-config.yaml`** - Pre-commit hooks for code quality
- **`tox.ini`** - Testing environments configuration

## 🚀 Automated Release Process

### 1. Create a Release

```bash
# Update version and create release
python release.py patch    # 0.1.0 → 0.1.1
python release.py minor    # 0.1.0 → 0.2.0
python release.py major    # 0.1.0 → 1.0.0

# Or set specific version
python release.py 1.0.0
```

### 2. Push to GitHub

```bash
git push origin main --tags
```

### 3. Automated Publishing

GitHub Actions will automatically:
1. **Build** wheels for multiple platforms (Linux, macOS, Windows)
2. **Test** wheels on different Python versions (3.10, 3.11, 3.12)
3. **Publish** to PyPI using trusted publishing
4. **Create** GitHub release with changelog

## 🔧 Local Development

### Install Development Dependencies

```bash
pip install -e .[dev]
```

### Run Quality Checks

```bash
# All checks
python build_script.py check

# Individual checks
python build_script.py lint    # Code formatting and linting
python build_script.py test    # Run tests with coverage

# Build locally
python build_script.py build

# See all available commands
python build_script.py --help
```

### Multi-Environment Testing

```bash
# Test with tox (requires Python 3.10, 3.11, 3.12 installed)
tox

# Test specific environment
tox -e py310
```

## 📦 Package Configuration

### pyproject.toml Highlights

- **Build system**: Hatchling (modern, fast)
- **Python versions**: 3.10, 3.11, 3.12
- **Dependencies**: NumPy, Pandas, Pydantic, SciPy
- **Optional dependencies**: dev, docs, performance extras
- **Entry points**: None (pure library)

### Wheel Building

The package builds **universal wheels** that work on:
- **Platforms**: Linux, macOS, Windows
- **Architectures**: x64, ARM64 (where available)
- **Python versions**: 3.10, 3.11, 3.12

## 🔐 PyPI Publishing

### Trusted Publishing

Uses GitHub's **Trusted Publishing** (OIDC) for secure PyPI uploads:
- No API tokens stored in repository
- Automatic authentication via GitHub Actions
- Scoped to specific repository only

### Release Workflow

1. **Tag pushed** → Release workflow triggers
2. **Build wheels** → Multi-platform compilation
3. **Test wheels** → Installation and import verification
4. **Publish to PyPI** → Upload source + wheels
5. **Create release** → GitHub release with auto-generated notes

## 🧪 Testing Strategy

### CI Pipeline
- **Unit tests** on Python 3.10, 3.11, 3.12
- **Code coverage** ≥70% requirement
- **Linting**: Black, Ruff, MyPy
- **Pre-commit hooks**: Automated quality checks

### Test Categories
- **Unit tests**: Individual functions and methods
- **Integration tests**: Model interactions
- **Validation tests**: Against reference datasets
- **Performance tests**: Benchmarks and profiling

## 📋 Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run
```

Hooks include:
- **Trailing whitespace** removal
- **End-of-file** fixer
- **YAML validation**
- **Black** code formatting
- **Ruff** linting
- **MyPy** type checking

## 🚨 Troubleshooting

### Build Issues

**Problem**: `ModuleNotFoundError` during testing
**Solution**: Ensure test dependencies are installed
```bash
pip install -e .[dev]
```

**Problem**: Import errors in built package
**Solution**: Check `src/` directory structure and `pyproject.toml` packages configuration

### Publishing Issues

**Problem**: PyPI upload fails
**Solution**: Check trusted publishing setup and repository permissions

**Problem**: Wheel build fails on specific platform
**Solution**: Check for platform-specific dependencies or compilation issues

### Version Issues

**Problem**: Version not updating in built package
**Solution**: Ensure `pyproject.toml` version is updated before building

## 📚 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Hatchling Documentation](https://hatch.pypa.io/latest/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/guides/building-and-testing-python)

## 🎯 Next Steps

1. **Test the CI pipeline** by pushing to a feature branch
2. **Configure PyPI trusted publishing** in repository settings
3. **Set up documentation building** (Read the Docs)
4. **Add performance testing** to CI pipeline
5. **Configure code coverage** reporting (Codecov)
