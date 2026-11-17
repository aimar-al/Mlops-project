"""Unit tests of the preprocessing logic (src/preprocessing.py).

Every functionality of the logic is tested in isolation. The ``numeric_list``
fixture provides a reusable list of numbers and ``pytest.mark.parametrize``
repeats several tests with different inputs and options.
"""

import math

import pytest

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


@pytest.fixture
def numeric_list() -> list[float]:
    """Provide a list of numbers shared by several tests."""
    return [2.0, 4.0, 6.0, 8.0]


def test_remove_missing_values():
    """None, NaN and empty or 'nan'/'none' strings are removed."""
    values = [1, None, "", "   ", float("nan"), "nan", "None", "data"]
    assert remove_missing_values(values) == [1, "data"]


def test_remove_missing_values_keeps_other_values():
    """Values that are not missing (even falsy ones) are kept."""
    assert remove_missing_values(["a", 0, False, 3.5]) == ["a", 0, False, 3.5]


@pytest.mark.parametrize(
    ("fill_value", "expected"),
    [
        (0, [0, 0, "a", 0]),
        ("x", ["x", "x", "a", "x"]),
        (None, [None, None, "a", None]),
    ],
)
def test_fill_missing_values(fill_value, expected):
    """The missing values are replaced with the given fill value."""
    values = [None, "", "a", float("nan")]
    assert fill_missing_values(values, fill_value=fill_value) == expected


def test_fill_missing_values_default():
    """Without a fill value, the missing values are replaced by 0."""
    assert fill_missing_values([None, 1, "", 2]) == [0, 1, 0, 2]


@pytest.mark.parametrize("values", [[3, 1, 3, 2, 1], ["b", "a", "b"], []])
def test_remove_duplicated_values(values):
    """Only the first occurrence of every value is kept, in order."""
    assert remove_duplicated_values(values) == list(dict.fromkeys(values))


def test_normalize_min_max_default_range(numeric_list):
    """With the default range, the minimum maps to 0 and the maximum to 1."""
    result = normalize_min_max(numeric_list)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0 / 3.0)
    assert result[2] == pytest.approx(2.0 / 3.0)
    assert result[3] == pytest.approx(1.0)


@pytest.mark.parametrize(("new_min", "new_max"), [(0.0, 1.0), (0.0, 10.0), (-5.0, 5.0), (1.0, 2.0)])
def test_normalize_min_max_ranges(numeric_list, new_min, new_max):
    """The values are scaled to the requested [new_min, new_max] range."""
    result = normalize_min_max(numeric_list, new_min=new_min, new_max=new_max)
    assert result[0] == pytest.approx(new_min)
    assert result[-1] == pytest.approx(new_max)


def test_normalize_min_max_constant_list():
    """A constant list maps every value to new_min (avoids dividing by 0)."""
    assert normalize_min_max([5.0, 5.0, 5.0], new_min=-1.0, new_max=1.0) == [-1.0, -1.0, -1.0]


def test_normalize_min_max_skips_non_numeric():
    """Values that do not represent a finite number are skipped."""
    assert normalize_min_max([1, "2", "abc", None]) == [0.0, 1.0]


def test_standardize_z_score_properties(numeric_list):
    """The standardized list has mean 0 and standard deviation 1."""
    result = standardize_z_score(numeric_list)
    mean = sum(result) / len(result)
    variance = sum(value**2 for value in result) / len(result)
    assert mean == pytest.approx(0.0)
    assert variance == pytest.approx(1.0)


def test_standardize_z_score_known_values():
    """A known list produces its exact standardized values."""
    result = standardize_z_score([1.0, 2.0, 3.0])
    assert result == pytest.approx([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])


def test_standardize_z_score_constant_list():
    """A constant list is mapped to zeros (the standard deviation is 0)."""
    assert standardize_z_score([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(("min_value", "max_value"), [(2.0, 8.0), (0.0, 1.0), (3.0, 3.0)])
def test_clip_values_ranges(numeric_list, min_value, max_value):
    """Every value ends inside the [min_value, max_value] range."""
    result = clip_values(numeric_list, min_value=min_value, max_value=max_value)
    for value in result:
        assert min_value <= value <= max_value


def test_clip_values_known_values():
    """A known list is clipped exactly as expected."""
    assert clip_values([1.0, 5.0, 10.0], min_value=2.0, max_value=8.0) == [2.0, 5.0, 8.0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", [3]),
        ("3.7", [3]),
        ("  5 ", [5]),
        ("abc", []),
        ("", []),
        (None, []),
        (7, [7]),
        (2.9, [2]),
    ],
)
def test_convert_to_integers_single(value, expected):
    """Each single value converts as documented (or is excluded)."""
    assert convert_to_integers([value]) == expected


def test_convert_to_integers_mixed():
    """Only the numerical values of a mixed list are converted and kept."""
    values = ["3", "-2", "3.7", "abc", "", 4, 2.9, None]
    assert convert_to_integers(values) == [3, -2, 3, 4, 2]


def test_log_transform(numeric_list):
    """The positive numbers are transformed with the natural logarithm."""
    result = log_transform(numeric_list)
    expected = [math.log(value) for value in numeric_list]
    assert result == pytest.approx(expected)


def test_log_transform_keeps_only_positive():
    """Zeros, negatives and non numerical values are excluded."""
    values = [1, math.e, 0, -2, "4", "nan", None, "abc"]
    assert log_transform(values) == pytest.approx([0.0, 1.0, math.log(4.0)])


@pytest.mark.parametrize(
    ("text_value", "expected"),
    [
        ("Hello, World! 123", ["hello", "world", "123"]),
        ("multiple    spaces", ["multiple", "spaces"]),
        ("no-alphanumeric-here", ["no", "alphanumeric", "here"]),
        ("", []),
    ],
)
def test_tokenize_text(text_value, expected):
    """The text is lower cased and split into alphanumeric tokens."""
    assert tokenize_text(text_value) == expected


@pytest.mark.parametrize(
    ("text_value", "expected"),
    [
        ("Hello, World! 123", "Hello World 123"),
        ("a1b2@c3", "a1b2c3"),
        ("tabs\tand\nnewlines", "tabsandnewlines"),
        ("", ""),
    ],
)
def test_select_alphanumeric(text_value, expected):
    """Only alphanumeric characters and spaces survive."""
    assert select_alphanumeric(text_value) == expected


def test_remove_stopwords():
    """The stop-words are removed from the lower cased text."""
    result = remove_stopwords("The cat IS on the mat", ["the", "is", "on"])
    assert result == "cat mat"


def test_remove_stopwords_case_insensitive():
    """The stop-words are compared ignoring their case."""
    assert remove_stopwords("The cat is here", ["THE", "Is"]) == "cat here"


def test_remove_stopwords_without_stopwords():
    """Without stop-words, the text is only lower cased."""
    assert remove_stopwords("Hello World", []) == "hello world"


@pytest.mark.parametrize(
    ("nested", "expected"),
    [
        ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ([["a", "b"], ["c"]], ["a", "b", "c"]),
        ([1, [2, [3, [4]]]], [1, 2, 3, 4]),
        ([], []),
    ],
)
def test_flatten_list(nested, expected):
    """Lists of lists (at any depth) are flattened into one list."""
    assert flatten_list(nested) == expected


def test_shuffle_values_is_permutation():
    """The shuffled list is a permutation of the original one."""
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    result = shuffle_values(values)
    assert sorted(result) == sorted(values)
    assert len(result) == len(values)


def test_shuffle_values_reproducible_with_seed():
    """The same seed always produces the same order."""
    values = list(range(20))
    first = shuffle_values(values, seed=42)
    second = shuffle_values(values, seed=42)
    assert first == second
    assert sorted(first) == values


def test_shuffle_values_different_seeds_differ():
    """Two different seeds produce (for this input) different orders."""
    values = list(range(10))
    assert shuffle_values(values, seed=1) != shuffle_values(values, seed=2)


def test_shuffle_values_does_not_modify_input():
    """The input list is not modified."""
    values = [1, 2, 3]
    shuffle_values(values, seed=0)
    assert values == [1, 2, 3]
