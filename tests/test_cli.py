"""Integration tests between the preprocessing logic and the CLI.

The tests use a shared ``CliRunner`` fixture (sent as an input argument to
every test) to invoke commands of the four groups (clean, numeric, text and
struct) and check that the CLI uses the logic correctly.
"""

import json

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.preprocessing import shuffle_values


@pytest.fixture
def runner() -> CliRunner:
    """Provide a CliRunner shared by all the CLI integration tests."""
    return CliRunner()


def _output_list(result) -> list:
    """Parse the JSON list printed by a CLI command."""
    return json.loads(result.output)


def test_cli_help_lists_groups(runner):
    """The general group documents the four command groups."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for group in ("clean", "numeric", "text", "struct"):
        assert group in result.output


def test_clean_remove_missing(runner):
    """The clean group removes missing values through the logic."""
    result = runner.invoke(cli, ["clean", "remove-missing", "1", "None", "", "nan", "data"])
    assert result.exit_code == 0
    assert _output_list(result) == ["1", "data"]


def test_clean_fill_missing_with_option(runner):
    """The fill-missing command replaces the missing values with --fill."""
    result = runner.invoke(cli, ["clean", "fill-missing", "1", "None", "2", "--fill", "9"])
    assert result.exit_code == 0
    assert _output_list(result) == ["1", 9, "2"]


def test_clean_fill_missing_default(runner):
    """Without --fill, the missing values are replaced by 0."""
    result = runner.invoke(cli, ["clean", "fill-missing", "1", "None", "2"])
    assert result.exit_code == 0
    assert _output_list(result) == ["1", 0, "2"]


def test_numeric_normalize_with_options(runner):
    """The normalize command scales VALUES to the [--min, --max] range."""
    result = runner.invoke(cli, ["numeric", "normalize", "1", "2", "3", "--min", "0", "--max", "10"])
    assert result.exit_code == 0
    assert _output_list(result) == [0.0, 5.0, 10.0]


def test_numeric_normalize_default_options(runner):
    """Without options, the normalize command uses the range [0, 1]."""
    result = runner.invoke(cli, ["numeric", "normalize", "1", "2", "3", "4"])
    assert result.exit_code == 0
    assert _output_list(result) == [0.0, pytest.approx(1.0 / 3.0), pytest.approx(2.0 / 3.0), 1.0]


def test_numeric_standardize(runner):
    """The standardize command returns the z-scores of VALUES."""
    result = runner.invoke(cli, ["numeric", "standardize", "1", "2", "3"])
    values = _output_list(result)
    assert values[0] == pytest.approx(-1.2247448713915892)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(1.2247448713915892)


def test_numeric_clip_with_options(runner):
    """The clip command limits VALUES to the [--min, --max] range."""
    result = runner.invoke(cli, ["numeric", "clip", "1", "5", "10", "--min", "2", "--max", "8"])
    assert result.exit_code == 0
    assert _output_list(result) == [2.0, 5.0, 8.0]


def test_numeric_to_int(runner):
    """The to-int command converts and excludes non numerical values."""
    result = runner.invoke(cli, ["numeric", "to-int", "--", "3", "-2", "3.7", "abc", "4"])
    assert result.exit_code == 0
    assert _output_list(result) == [3, -2, 3, 4]


def test_numeric_log(runner):
    """The log command transforms only the positive values of VALUES."""
    result = runner.invoke(cli, ["numeric", "log", "--", "1", "2.718281828459045", "0", "-1"])
    values = _output_list(result)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)


def test_text_tokenize(runner):
    """The tokenize command splits TEXT_VALUE into alphanumeric tokens."""
    result = runner.invoke(cli, ["text", "tokenize", "Hello, World! 123"])
    assert result.exit_code == 0
    assert _output_list(result) == ["hello", "world", "123"]


def test_text_remove_punctuation(runner):
    """The remove-punctuation command keeps alphanumerics and spaces."""
    result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World!"])
    assert result.exit_code == 0
    assert result.output.strip() == "Hello World"


def test_text_remove_stopwords_with_option(runner):
    """The remove-stopwords command removes the comma separated stop-words."""
    result = runner.invoke(
        cli,
        ["text", "remove-stopwords", "The cat is on the mat", "--stopwords", "the,is,on"],
    )
    assert result.exit_code == 0
    assert result.output.strip() == "cat mat"


def test_struct_shuffle_matches_logic(runner):
    """The shuffle command output equals the logic output for the same seed."""
    result = runner.invoke(cli, ["struct", "shuffle", "1", "2", "3", "4", "--seed", "42"])
    assert result.exit_code == 0
    expected = shuffle_values(["1", "2", "3", "4"], seed=42)
    assert _output_list(result) == expected


def test_struct_flatten(runner):
    """The flatten command flattens the comma separated sublists."""
    result = runner.invoke(cli, ["struct", "flatten", "1,2,3", "4,5"])
    assert result.exit_code == 0
    assert _output_list(result) == ["1", "2", "3", "4", "5"]


def test_struct_unique(runner):
    """The unique command keeps the first occurrence of every value."""
    result = runner.invoke(cli, ["struct", "unique", "3", "1", "3", "2", "1"])
    assert result.exit_code == 0
    assert _output_list(result) == ["3", "1", "2"]


def test_numeric_normalize_rejects_non_numeric(runner):
    """Non numeric values for a typed argument produce a usage error."""
    result = runner.invoke(cli, ["numeric", "normalize", "1", "abc"])
    assert result.exit_code != 0
