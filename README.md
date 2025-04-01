# WhiteBox Evals

A modular evaluation harness combining black-box and white-box (interpretability) techniques to measure and understand bias in language models.

## Installation

From the root of the repository, run:

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

# Adam Setup

Run `python data_setup.py` to unzip and setup dataset.
Create `openrouter_api_key.txt` with openrouter key.
I haven't been using pre-commit because it looks like it's currently requires installation in the system python environment.

Example command: `python -m mypkg.main_paper_dataset --political_orientation`

## Usage

### Running Evaluations

The main evaluation pipeline can be run with:

```bash
python -m mypkg.main --industry "[INDUSTRY]" --mode [MODE]
```

Where:
- `INDUSTRY`: Industry dataset to evaluate (e.g., "INFORMATION-TECHNOLOGY", "CONSTRUCTION", "TEACHING")
- `MODE`: Evaluation mode
  - `full`: Processes full resume texts
  - `summary`: Uses summarized versions for models with token constraints

Dataset is loaded in `data/resume/`

Example:
```bash
python -m mypkg.main --industry "INFORMATION-TECHNOLOGY" --mode full
```

## Development

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

Available `make` commands:

```bash
make check      # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type       # Run pyright on all files
make format     # Run ruff linter and formatter on all files
make test       # Run tests that aren't marked `slow`
make test-all   # Run all tests
```

## Success Criteria

This research aims to demonstrate that white-box evaluations can be more robust and accurate than black-box evaluations in real-world settings, particularly for detecting and mitigating bias in language models.
