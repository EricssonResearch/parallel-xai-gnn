# Declare all phony targets
.PHONY: pipeline

# Define pipeline
pipeline:
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@black --check ./
	@mypy ./
	@flake8 ./
	@pylint ./