"""
============================= test session starts ==============================
platform darwin -- Python 3.11.8, pytest-7.4.4, pluggy-1.0.0
cachedir: .pytest_cache
rootdir: .
plugins: cov-4.1.0
collecting ... collected 5 items

test_assignta.py::test_overallocation PASSED                             [ 20%]
test_assignta.py::test_conflicts PASSED                                  [ 40%]
test_assignta.py::test_undersupport PASSED                               [ 60%]
test_assignta.py::test_unwilling PASSED                                  [ 80%]
test_assignta.py::test_unpreferred PASSED                                [100%]

---------- coverage: platform darwin, python 3.11.8-final-0 ----------
Name               Stmts   Miss  Cover   Missing
------------------------------------------------
assignta.py          166     78    53%   120-129, 134-146, 151-172, 177-189, 193-234, 237
evo.py                85     57    33%   13-15, 19, 25, 29-30, 34-39, 44-59, 67-70, 74, 78-79, 83-101, 105-108, 113-114, 119-137
profiler.py           36      8    78%   27, 39-40, 45-49
test_assignta.py      63      0   100%
------------------------------------------------
TOTAL                350    143    59%


============================== 5 passed in 3.25s ===============================
"""

import pandas as pd
import pytest
import numpy as np

from assignta import overallocation, conflicts, undersupport, unwilling, unpreferred
from assignta import load_tas, load_sections

# Load sections and TAs
sections = load_sections('sections.csv')
tas = load_tas('tas.csv')

# Load test cases
test1 = pd.read_csv('test1.csv', header = None)
test2 = pd.read_csv('test2.csv', header = None)
test3 = pd.read_csv('test3.csv', header = None)

assignments1 = test1.values
assignments2 = test2.values
assignments3 = test3.values

def test_overallocation():
    # Test Case 1
    overallocation_penalty = overallocation(assignments1, tas)
    expected_overallocation_1 = 37
    assert overallocation_penalty == expected_overallocation_1, f"Expected overallocation penalty {expected_overallocation_1} but got {overallocation_penalty} for test1.csv"

    # Test Case 2
    overallocation_penalty = overallocation(assignments2, tas)
    expected_overallocation_2 = 41
    assert overallocation_penalty == expected_overallocation_2, f"Expected overallocation penalty {expected_overallocation_2} but got {overallocation_penalty} for test2.csv"

    # Test Case 3
    overallocation_penalty = overallocation(assignments3, tas)
    expected_overallocation_3 = 23
    assert overallocation_penalty == expected_overallocation_3, f"Expected overallocation penalty {expected_overallocation_3} but got {overallocation_penalty} for test3.csv"

def test_conflicts():
    # Test Case 1
    conflict_penalty = conflicts(assignments1, sections)
    expected_conflicts_1 = 8
    assert conflict_penalty == expected_conflicts_1, f"Expected conflict penalty {expected_conflicts_1} but got {conflict_penalty} for test1.csv"

    # Test Case 2
    conflict_penalty = conflicts(assignments2, sections)
    expected_conflicts_2 = 5
    assert conflict_penalty == expected_conflicts_2, f"Expected conflict penalty {expected_conflicts_2} but got {conflict_penalty} for test2.csv"

    # Test Case 3
    conflict_penalty = conflicts(assignments3, sections)
    expected_conflicts_3 = 2
    assert conflict_penalty == expected_conflicts_3, f"Expected conflict penalty {expected_conflicts_3} but got {conflict_penalty} for test3.csv"

def test_undersupport():
    # Test Case 1
    undersupport_penalty = undersupport(assignments1, sections, tas)
    expected_undersupport_1 = 1
    assert undersupport_penalty == expected_undersupport_1, f"Expected undersupport penalty {expected_undersupport_1} but got {undersupport_penalty} for test1.csv"

    # Test Case 2
    undersupport_penalty = undersupport(assignments2, sections, tas)
    expected_undersupport_2 = 0
    assert undersupport_penalty == expected_undersupport_2, f"Expected undersupport penalty {expected_undersupport_2} but got {undersupport_penalty} for test2.csv"

    # Test Case 3
    undersupport_penalty = undersupport(assignments3, sections, tas)
    expected_undersupport_3 = 7
    assert undersupport_penalty == expected_undersupport_3, f"Expected undersupport penalty {expected_undersupport_3} but got {undersupport_penalty} for test3.csv"

def test_unwilling():
    # Test Case 1
    unwilling_penalty = unwilling(assignments1, tas)
    expected_unwilling_1 = 53
    assert unwilling_penalty == expected_unwilling_1, f"Expected unwillingness penalty {expected_unwilling_1} but got {unwilling_penalty} for test1.csv"

    # Test Case 2
    unwilling_penalty = unwilling(assignments2, tas)
    expected_unwilling_2 = 58
    assert unwilling_penalty == expected_unwilling_2, f"Expected unwillingness penalty {expected_unwilling_2} but got {unwilling_penalty} for test2.csv"

    # Test Case 3
    unwilling_penalty = unwilling(assignments3, tas)
    expected_unwilling_3 = 43
    assert unwilling_penalty == expected_unwilling_3, f"Expected unwillingness penalty {expected_unwilling_3} but got {unwilling_penalty} for test3.csv"

def test_unpreferred():
    # Test Case 1
    unpreferred_penalty = unpreferred(assignments1, tas)
    expected_unpreferred_1 = 15
    assert unpreferred_penalty == expected_unpreferred_1, f"Expected unpreferred penalty {expected_unpreferred_1} but got {unpreferred_penalty} for test1.csv"

    # Test Case 2
    unpreferred_penalty = unpreferred(assignments2, tas)
    expected_unpreferred_2 = 19
    assert unpreferred_penalty == expected_unpreferred_2, f"Expected unpreferred penalty {expected_unpreferred_2} but got {unpreferred_penalty} for test2.csv"

    # Test Case 3
    unpreferred_penalty = unpreferred(assignments3, tas)
    expected_unpreferred_3 = 10
    assert unpreferred_penalty == expected_unpreferred_3, f"Expected unpreferred penalty {expected_unpreferred_3} but got {unpreferred_penalty} for test3.csv"

# python -m pytest -v --cov --cov-report term-missing > unit_test_report.txt