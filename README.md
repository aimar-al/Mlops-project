# Lab0 — Data Preprocessing with CI Fundamentals

Project for the **MLOps** course (Master's degree in Machine Learning).
It implements a set of data preprocessing techniques (logic), a command
line interface (CLI) to use them, and a test suite (unit and integration
tests) that checks both components.

## Project structure

```
Lab0/
├── src/
│   ├── preprocessing.py   # logic: the preprocessing functions
│   ├── cli.py             # CLI with click (group cli: clean, numeric, text, struct)
│   └── __init__.py
├── tests/
│   ├── test_preprocessing.py   # unit tests of the logic
│   └── test_cli.py             # integration tests (CLI <-> logic, CliRunner)
├── pyproject.toml         # dependencies and tool configuration
├── .python-version        # fixed Python version (3.13)
└── README.md
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) as package and
environment manager:

```bash
uv sync          # creates .venv and installs the dependencies from uv.lock
```

Runtime dependency: `click`. Development dependencies (installed with the
default `uv sync`, they would be added with `uv add --dev`): `black`,
`pylint`, `pytest`, `pytest-cov`.

## Usage

The CLI has a general group `cli` with four subgroups, one command per
function of the logic:

```bash
uv run python -m src.cli --help
uv run python -m src.cli clean remove-missing 1 None nan data
uv run python -m src.cli clean fill-missing 1 None nan 3 --fill 9
uv run python -m src.cli numeric normalize 1 2 3 4 5 --min 0 --max 10
uv run python -m src.cli numeric standardize 2 4 6 8
uv run python -m src.cli numeric clip 1 5 10 --min 2 --max 8
uv run python -m src.cli numeric to-int -- 3 -2 3.7 abc 4
uv run python -m src.cli numeric log 1 2.718281828459045 10
uv run python -m src.cli text tokenize "Hello, World! 123"
uv run python -m src.cli text remove-punctuation "Hello, World!"
uv run python -m src.cli text remove-stopwords "The cat is on the mat" --stopwords "the,is,on"
uv run python -m src.cli struct shuffle 1 2 3 4 --seed 42
uv run python -m src.cli struct flatten 1,2,3 4,5
uv run python -m src.cli struct unique 3 1 3 2 1
```

Notes:

- Lists are passed as several arguments (one value per argument); `flatten`
  receives one sublist per argument, with its items separated by commas.
- A value is treated as missing when it is `None`, NaN, an empty string, or
  the string `"nan"`/`"none"` (ignoring case and surrounding spaces).
- `remove-punctuation` is implemented by the logic function
  `select_alphanumeric` (it keeps alphanumeric characters and spaces).
- Values that start with `-` (for example negative numbers) need `--`
  before them so that they are not taken as options.

## Quality checks

```bash
uv run python -m pylint src/*.py             # linting
uv run black src/*.py                        # formatting
uv run python -m pytest -v                   # testing
uv run python -m pytest -v --cov=src         # testing with coverage
```

## Testing strategy

- **Unit tests** (`tests/test_preprocessing.py`): one test group per logic
  function, including edge cases (empty lists, constant lists, NaN, non
  numerical values, reproducibility of the shuffle with a seed). A shared
  `numeric_list` fixture is reused by several tests, and
  `pytest.mark.parametrize` is used both in functionalities without
  options (fill missing, remove duplicated) and with options (normalize
  ranges, clip ranges, conversions).
- **Integration tests** (`tests/test_cli.py`): a shared `CliRunner`
  fixture invokes commands of the four groups and checks the output of the
  CLI against the expected logic results (for instance, the shuffle of the
  CLI is compared with the same shuffle executed by the logic).
