"""Run average-rank arithmetic checks without importing a local ML environment."""

from __future__ import annotations

import ast
import itertools
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/stock_forecasting/metrics.py"


class _Vector:
    """Small list-backed array adapter for the extracted production function."""

    def __init__(self, values):
        self.values = list(values)

    @property
    def size(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        if isinstance(index, _Vector):
            return _Vector(self.values[i] for i in index)
        result = self.values[index]
        return _Vector(result) if isinstance(index, slice) else result

    def __setitem__(self, indices, values):
        for index, value in zip(indices, values, strict=True):
            self.values[index] = value

    def _binary(self, other, operation):
        right = other if isinstance(other, _Vector) else [other] * self.size
        return _Vector(operation(a, b) for a, b in zip(self, right, strict=True))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __ne__(self, other):
        return self._binary(other, lambda a, b: a != b)


class _ArrayPrimitives:
    float64 = float

    @staticmethod
    def asarray(values):
        return _Vector(values)

    @staticmethod
    def empty(size, dtype):
        return _Vector(dtype(0) for _ in range(size))

    @staticmethod
    def argsort(values, *, kind):
        if kind != "stable":
            raise AssertionError("The production ranking must use a stable sort")
        return _Vector(sorted(range(values.size), key=lambda i: values[i]))

    @staticmethod
    def flatnonzero(values):
        return _Vector(i for i, value in enumerate(values) if value)

    @staticmethod
    def concatenate(arrays):
        return _Vector(itertools.chain.from_iterable(arrays))

    @staticmethod
    def repeat(values, counts):
        return _Vector(
            value for value, count in zip(values, counts, strict=True) for _ in range(count)
        )


def production_rank():
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if getattr(node, "name", "") == "_rank")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    namespace = {"np": _ArrayPrimitives}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_rank"]


class AverageRankControlTests(unittest.TestCase):
    def test_manual_tied_rank_correlation_matches_five_sixths(self):
        rank = production_rank()
        target = list(rank([1, 1, 2, 3]))
        signal = list(rank([1, 2, 2, 4]))
        self.assertEqual(target, [0.5, 0.5, 2.0, 3.0])
        self.assertEqual(signal, [0.0, 1.5, 1.5, 3.0])
        centered_target = [value - sum(target) / len(target) for value in target]
        centered_signal = [value - sum(signal) / len(signal) for value in signal]
        numerator = sum(a * b for a, b in zip(centered_target, centered_signal, strict=True))
        target_norm = sum(value * value for value in centered_target)
        signal_norm = sum(value * value for value in centered_signal)
        self.assertEqual((numerator, target_norm, signal_norm), (3.75, 4.5, 4.5))
        self.assertAlmostEqual(numerator / math.sqrt(target_norm * signal_norm), 5 / 6)

    def test_production_average_ranks_include_ties_negatives_and_empty_input(self):
        rank = production_rank()
        for values, expected in (
            ([], []),
            ([7], [0.0]),
            ([3, 1, 2], [2.0, 0.0, 1.0]),
            ([4, 4, 4], [1.0, 1.0, 1.0]),
            ([10, 20, 20, -3, -3], [2.0, 3.5, 3.5, 0.5, 0.5]),
        ):
            with self.subTest(values=values):
                self.assertEqual(list(rank(values)), expected)

    def test_ties_do_not_depend_on_symbol_input_order(self):
        rank = production_rank()
        values = [1, 1, 2, 4, 4]
        expected_by_value = {1: 0.5, 2: 2.0, 4: 3.5}
        for permutation in itertools.permutations(values):
            self.assertEqual(list(rank(permutation)), [expected_by_value[v] for v in permutation])

    def test_unique_values_preserve_the_previous_zero_based_ranking(self):
        rank = production_rank()
        for permutation in itertools.permutations(range(5)):
            self.assertEqual(list(rank(permutation)), list(permutation))


if __name__ == "__main__":
    unittest.main()
