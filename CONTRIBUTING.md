# Contributing to HyperCrystal

Thank you for your interest in HyperCrystal! We welcome contributions.

## Development Setup

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/hypercrystal.git
   cd hypercrystal
   ```
3. Run the setup script:
   ```bash
   ./scripts/init_setup.sh
   source .venv/bin/activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Use `black` for formatting: `black hypercrystal/`
- Use `isort` for import sorting: `isort hypercrystal/`
- Write docstrings for public functions.

## Testing

Run tests with:
```bash
pytest hypercrystal/tests/
```

## Pull Request Process

1. Ensure your code passes existing tests.
2. Update documentation if needed.
3. Create a PR against the `main` branch.
4. Request review from maintainers.

## Reporting Issues

Use GitHub Issues. Include:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
