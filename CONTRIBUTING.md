# Contributing to CVD Risk Calculator

Thank you for your interest in contributing to CVD Risk Calculator! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/m-aljasem/PyCVDRisk.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install in development mode: `pip install -e ".[dev]"`

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Check code quality: `black src tests && ruff check src tests && mypy src`
5. Commit with clear messages: `git commit -m "Add feature X"`
6. Push and create a pull request

## Coding Standards

- **Type Hints**: All functions must have complete type hints
- **Docstrings**: Use NumPy-style docstrings for all public functions/classes
- **Testing**: Maintain 95%+ test coverage
- **Formatting**: Code is automatically formatted with Black and Ruff

## Adding New Risk Models

1. Create a new file in `src/cvd_risk_calculator/models/`
2. Inherit from `RiskModel` base class
3. Implement `calculate()` method
4. Add comprehensive tests in `tests/unit/test_<model_name>.py`
5. Create a Jupyter notebook in `notebooks/<model_name>/`
6. Update documentation

## Testing

Run tests with:
```bash
pytest                    # All tests
pytest tests/unit/        # Unit tests only
pytest --cov             # With coverage
pytest -v                # Verbose output
```

## Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Update CHANGELOG.md
4. Request review from maintainers
5. Address review comments

## Questions?

Open an issue for questions, bug reports, or feature requests.

