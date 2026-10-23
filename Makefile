# Declare all phony targets
.PHONY: new attach ruff ty isort uv_sort check

# Allow `make new <session>` / `make attach <session>`: take the 2nd word as the
# session name and turn it into a no-op target so make doesn't try to build it.
# `make new name=<session>` still works too.
ifneq ($(filter $(firstword $(MAKECMDGOALS)),new attach),)
  name ?= $(word 2,$(MAKECMDGOALS))
  ifneq ($(name),)
    $(eval $(name):;@:)
  endif
endif

new:
	@tmux new-session -s $(name)

attach:
	@test -n "$(name)" || (echo "Usage: make attach <session>"; exit 1)
	@tmux attach -t $(name)

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
