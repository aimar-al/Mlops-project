"""Command line interface (CLI) of the Lab0 preprocessing project.

The interface is built with Click: a general group named ``cli`` with four
subgroups (``clean``, ``numeric``, ``text`` and ``struct``), one command per
function of the preprocessing logic. Every command prints its result as
JSON (lists) or plain text (processed strings).

Run it from the project root, for example:

    uv run python -m src.cli --help
    uv run python -m src.cli clean remove-missing 1 None nan data
"""

import json

import click

from src.preprocessing import (
    clip_values,
    convert_to_integers,
    fill_missing_values,
    flatten_list,
    log_transform,
    normalize_min_max,
    remove_duplicated_values,
    remove_missing_values,
    remove_stopwords,
    select_alphanumeric,
    shuffle_values,
    standardize_z_score,
    tokenize_text,
)


def _echo_result(result: object) -> None:
    """Print ``result``: plain text for strings, JSON for anything else."""
    if isinstance(result, str):
        click.echo(result)
    else:
        click.echo(json.dumps(result, ensure_ascii=False, default=str))


@click.group()
def cli() -> None:
    """Preprocessing CLI with clean, numeric, text and struct commands."""


@cli.group()
def clean() -> None:
    """Commands to clean data: missing and duplicated values."""


@clean.command("remove-missing")
@click.argument("values", nargs=-1)
def remove_missing_command(values: tuple[str, ...]) -> None:
    """Remove the missing values (None, empty strings, nan) of VALUES.

    Example: uv run python -m src.cli clean remove-missing 1 None nan data
    """
    _echo_result(remove_missing_values(list(values)))


@clean.command("fill-missing")
@click.argument("values", nargs=-1)
@click.option("--fill", default=0, help="Value that replaces the missing ones (default: 0).")
def fill_missing_command(values: tuple[str, ...], fill: object) -> None:
    """Replace the missing values of VALUES with the value of --fill.

    Example: uv run python -m src.cli clean fill-missing 1 None nan 3 --fill 9
    """
    _echo_result(fill_missing_values(list(values), fill_value=fill))


@cli.group()
def numeric() -> None:
    """Commands for numerical attributes."""


@numeric.command("normalize")
@click.argument("values", nargs=-1, type=float)
@click.option("--min", "new_min", default=0.0, help="Minimum of the new range (default: 0).")
@click.option("--max", "new_max", default=1.0, help="Maximum of the new range (default: 1).")
def normalize_command(values: tuple[float, ...], new_min: float, new_max: float) -> None:
    """Normalize VALUES to the [--min, --max] range (min-max method).

    Example: uv run python -m src.cli numeric normalize 1 2 3 4 5 --min 0 --max 10
    """
    _echo_result(normalize_min_max(list(values), new_min=new_min, new_max=new_max))


@numeric.command("standardize")
@click.argument("values", nargs=-1, type=float)
def standardize_command(values: tuple[float, ...]) -> None:
    """Standardize VALUES with the z-score method.

    Example: uv run python -m src.cli numeric standardize 2 4 6 8
    """
    _echo_result(standardize_z_score(list(values)))


@numeric.command("clip")
@click.argument("values", nargs=-1, type=float)
@click.option("--min", "min_value", default=0.0, help="Minimum value to clip (default: 0).")
@click.option("--max", "max_value", default=1.0, help="Maximum value to clip (default: 1).")
def clip_command(values: tuple[float, ...], min_value: float, max_value: float) -> None:
    """Clip VALUES to the [--min, --max] range.

    Example: uv run python -m src.cli numeric clip 1 5 10 --min 2 --max 8
    """
    _echo_result(clip_values(list(values), min_value=min_value, max_value=max_value))


@numeric.command("to-int")
@click.argument("values", nargs=-1)
def to_int_command(values: tuple[str, ...]) -> None:
    """Convert the numerical values of VALUES to integers.

    Non numerical values are excluded from the output. Use ``--`` before
    the values when one of them starts with ``-`` (negative numbers).

    Example: uv run python -m src.cli numeric to-int -- 3 -2 3.7 abc 4
    """
    _echo_result(convert_to_integers(list(values)))


@numeric.command("log")
@click.argument("values", nargs=-1, type=float)
def log_command(values: tuple[float, ...]) -> None:
    """Transform the strictly positive values of VALUES to log scale.

    Example: uv run python -m src.cli numeric log 1 2.718281828459045 10
    """
    _echo_result(log_transform(list(values)))


@cli.group()
def text() -> None:
    """Commands to preprocess textual information."""


@text.command("tokenize")
@click.argument("text_value")
def tokenize_command(text_value: str) -> None:
    """Split TEXT_VALUE into lower cased alphanumeric word tokens.

    Example: uv run python -m src.cli text tokenize "Hello, World! 123"
    """
    _echo_result(tokenize_text(text_value))


@text.command("remove-punctuation")
@click.argument("text_value")
def remove_punctuation_command(text_value: str) -> None:
    """Remove the punctuation of TEXT_VALUE (only alphanumerics and spaces).

    Example: uv run python -m src.cli text remove-punctuation "Hello, World!"
    """
    _echo_result(select_alphanumeric(text_value))


@text.command("remove-stopwords")
@click.argument("text_value")
@click.option("--stopwords", default="", help='Comma separated stop-words, e.g. "the,is".')
def remove_stopwords_command(text_value: str, stopwords: str) -> None:
    """Remove the comma separated --stopwords from the lower cased TEXT_VALUE.

    Example: uv run python -m src.cli text remove-stopwords "The cat is here" --stopwords "the,is"
    """
    stop_list = [word.strip().lower() for word in stopwords.split(",") if word.strip()]
    _echo_result(remove_stopwords(text_value, stop_list))


@cli.group()
def struct() -> None:
    """Commands for the structure of the data."""


@struct.command("shuffle")
@click.argument("values", nargs=-1)
@click.option("--seed", type=int, default=None, help="Seed for a reproducible shuffle.")
def shuffle_command(values: tuple[str, ...], seed: int | None) -> None:
    """Shuffle VALUES; with --seed the result is reproducible.

    Example: uv run python -m src.cli struct shuffle 1 2 3 4 --seed 42
    """
    _echo_result(shuffle_values(list(values), seed=seed))


@struct.command("flatten")
@click.argument("sublists", nargs=-1)
def flatten_command(sublists: tuple[str, ...]) -> None:
    """Flatten SUBLISTS into one list (items of a sublist separated by commas).

    Example: uv run python -m src.cli struct flatten 1,2,3 4,5
    """
    nested = [sublist.split(",") for sublist in sublists]
    _echo_result(flatten_list(nested))


@struct.command("unique")
@click.argument("values", nargs=-1)
def unique_command(values: tuple[str, ...]) -> None:
    """Remove the duplicated values of VALUES keeping the first occurrences.

    Example: uv run python -m src.cli struct unique 3 1 3 2 1
    """
    _echo_result(remove_duplicated_values(list(values)))


if __name__ == "__main__":
    cli()
