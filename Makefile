# Declare all phony targets
.PHONY: ruff ty isort uv_sort check

format:
	@echo "Formatting code..."
	@uv run ruff format .
	@echo "✅ Code formatted."

ty:
	@echo "Checking typing..."
	@uv run ty check .
	@echo "✅ Types checked."

isort:
	@echo "Sorting imports..."
	@uv run ruff check --select I --fix .
	@echo "✅ Sorting Completed."

uv_sort:
	@echo "Sorting dependencies..."
	@uv run uv-sort pyproject.toml
	@echo "✅ Sorting dependencies completed."

check:
	@echo "Executing check..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@uv run ruff format --check ./
	@uv run ruff check .
	@uv run ty check .
	@echo "✅ Code Check Passed."
