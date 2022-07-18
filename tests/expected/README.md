# Deep Continuation tests

## Install requirements

Install tests requirements as

    pip install pytest pytest-mock

## Usage

Run the tests from the project's root directory

    pytest

## Reset the expected files

When finding a fundamental bug in how the data is produce, the expected output of the tests should change. The expected output of the test cases can be generated as follows

    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 55555 --fixstd 8.86
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 555 --fixstd 8.86
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 1 --fixstd 8.86
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 55555
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 555
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 1
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 55555 --fixstd [5,15]
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 555 --fixstd [5,15]
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --seed 1 --fixstd [5,15]
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 55555 --fixstd 8.86
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 555 --fixstd 8.86
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 1 --fixstd 8.86
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 55555
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 555
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 1
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 55555 --fixstd [5,15]
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 555 --fixstd [5,15]
    python deep_continuation/dataset.py --path tests/expected/ --size 4 --beta [0,60] --seed 1 --fixstd [5,15]
