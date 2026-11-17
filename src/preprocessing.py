"""Data preprocessing logic for the Lab0 project.

The module contains small, pure functions to clean lists of values and to
preprocess text. Every function is side effect free (it never modifies its
inputs), which makes the logic easy to test and to reuse from the CLI.
"""

import math
import random
import re

# Strings treated as missing values (compared ignoring case and surrounding
# spaces), so that values coming from the command line behave like None/nan.
MISSING_STRINGS = ("", "nan", "none")


def _is_missing(value: object) -> bool:
    """Return ``True`` when ``value`` is considered a missing value."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        return True
    return False


def remove_missing_values(values: list) -> list:
    """Remove the missing values of ``values``.

    A value is missing when it is ``None``, a NaN number, an empty or
    whitespace only string, or a string equal to ``"nan"`` or ``"none"``
    ignoring case and surrounding spaces.

    :param values: list of values that may contain missing entries.
    :return: list with the missing values removed.
    """
    return [value for value in values if not _is_missing(value)]


def fill_missing_values(values: list, fill_value: object = 0) -> list:
    """Replace the missing values of ``values`` with ``fill_value``.

    The missing values are the same that :func:`remove_missing_values`
    detects. ``fill_value`` defaults to ``0``.

    :param values: list of values that may contain missing entries.
    :param fill_value: value used to replace the missing ones (default 0).
    :return: list with the missing values replaced.
    """
    return [fill_value if _is_missing(value) else value for value in values]


def remove_duplicated_values(values: list) -> list:
    """Remove the duplicated values of ``values`` keeping the first occurrence.

    The comparison uses ``==``, so ``2`` and ``2.0`` are considered the same
    value. The order of the first occurrences is preserved.

    :param values: list of values that may contain duplicates.
    :return: list with the unique values in their original order.
    """
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _to_floats(values: list) -> list[float]:
    """Convert the values of ``values`` to ``float``, skipping the rest.

    A value is skipped when it does not represent a number or when the
    number is not finite (NaN and infinity are not usable in arithmetic).
    """
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def normalize_min_max(values: list, new_min: float = 0.0, new_max: float = 1.0) -> list[float]:
    """Scale the numeric values to the ``[new_min, new_max]`` range.

    The minimum of the values is mapped to ``new_min`` and the maximum to
    ``new_max``. Values that do not represent a finite number are skipped.
    When all the values are equal, every value is mapped to ``new_min``.

    :param values: list of values to normalize.
    :param new_min: minimum of the new range (default 0.0).
    :param new_max: maximum of the new range (default 1.0).
    :return: list with the normalized values.
    """
    numbers = _to_floats(values)
    if not numbers:
        return []
    smallest = min(numbers)
    span = max(numbers) - smallest
    normalized = []
    for number in numbers:
        if span == 0:
            normalized.append(new_min)
        else:
            scaled = (number - smallest) / span * (new_max - new_min) + new_min
            normalized.append(scaled)
    return normalized


def standardize_z_score(values: list) -> list[float]:
    """Standardize the numeric values with the z-score method.

    Each value is replaced by ``(value - mean) / std``, using the population
    standard deviation. Values that do not represent a finite number are
    skipped. When the standard deviation is 0 (all the values are equal),
    every value is mapped to ``0.0``.

    :param values: list of values to standardize.
    :return: list with the standardized values.
    """
    numbers = _to_floats(values)
    if not numbers:
        return []
    mean = sum(numbers) / len(numbers)
    variance = sum((number - mean) ** 2 for number in numbers) / len(numbers)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 for _ in numbers]
    return [(number - mean) / std for number in numbers]


def clip_values(values: list, min_value: float, max_value: float) -> list[float]:
    """Limit the numeric values to the ``[min_value, max_value]`` range.

    Values below ``min_value`` become ``min_value`` and values above
    ``max_value`` become ``max_value``. Values that do not represent a
    finite number are skipped.

    :param values: list of values to clip.
    :param min_value: minimum value of the range.
    :param max_value: maximum value of the range.
    :return: list with the clipped values.
    """
    clipped = []
    for number in _to_floats(values):
        clipped.append(max(min_value, min(number, max_value)))
    return clipped


def convert_to_integers(values: list) -> list[int]:
    """Convert the values of ``values`` that represent numbers to integers.

    Strings are converted with ``int`` or, when they represent a decimal
    number, by truncating the float (for example ``"3.7"`` becomes ``3``).
    Integers are kept as they are and floats are truncated. Non numerical
    values are excluded from the output.

    :param values: list of values (numerical and non numerical).
    :return: list with the values converted to integers.
    """
    integers = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            integers.append(value)
        elif isinstance(value, float):
            if math.isfinite(value):
                integers.append(int(value))
        elif isinstance(value, str):
            text = value.strip()
            try:
                integers.append(int(text))
            except ValueError:
                try:
                    integers.append(int(float(text)))
                except (ValueError, OverflowError):
                    continue
    return integers


def log_transform(values: list) -> list[float]:
    """Transform the positive numeric values of ``values`` to log scale.

    Only the values that represent a strictly positive finite number are
    kept, so zeros, negatives and non numerical values are excluded from
    the output.

    :param values: list of values to transform.
    :return: list with the natural logarithms of the positive values.
    """
    logarithms = []
    for number in _to_floats(values):
        if number > 0:
            logarithms.append(math.log(number))
    return logarithms


def tokenize_text(text: str) -> list[str]:
    """Split ``text`` into word tokens of alphanumeric characters.

    The text is lower cased and split into words; only the alphanumeric
    characters of each word are kept, so punctuation and other symbols are
    dropped (for example ``"Hello, World!"`` gives ``["hello", "world"]``).

    :param text: text to process.
    :return: list with the lower cased alphanumeric tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def select_alphanumeric(text: str) -> str:
    """Keep only the alphanumeric characters and the spaces of ``text``.

    Any other character (punctuation, symbols, tabs, newlines...) is
    removed. This function is the logic behind the ``remove-punctuation``
    command of the CLI.

    :param text: text to process.
    :return: text with only alphanumeric characters and spaces.
    """
    return re.sub(r"[^a-zA-Z0-9 ]", "", text)


def remove_stopwords(text: str, stopwords: list[str]) -> str:
    """Remove the ``stopwords`` from the lower cased ``text``.

    The text is lower cased and split into words; the words that appear in
    ``stopwords`` (compared ignoring case) are removed and the remaining
    ones are joined with single spaces.

    :param text: text to process.
    :param stopwords: stop-words to remove from the text.
    :return: processed text without the stop-words.
    """
    stop_set = {word.lower() for word in stopwords}
    words = [word for word in text.lower().split() if word not in stop_set]
    return " ".join(words)


def flatten_list(nested: list) -> list:
    """Flatten a list of lists into a single list.

    The flattening is recursive, so lists inside lists at any depth are
    also flattened. Values that are not lists are kept as they are.

    :param nested: list of lists to flatten.
    :return: flattened list with all the values.
    """
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def shuffle_values(values: list, seed: int | None = None) -> list:
    """Shuffle ``values`` and return the result in a new list.

    The input list is never modified. When ``seed`` is given, the shuffle
    is reproducible: the same values with the same seed always produce the
    same order.

    :param values: list of values to shuffle.
    :param seed: seed for reproducible shuffling (default ``None``).
    :return: new list with the values shuffled.
    """
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return shuffled
