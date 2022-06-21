# Deep Continuation tests

## Install requirements

Install tests requirements as

    pip install pytest pytest-mock

## Usage

Run the tests from the project's root directory

    pytest

## Reset the expected files

When finding a fundamental bug in how the data is produce, the expected output of the tests should change. They can be overwritten as follows

    python deep_continuation/dataset.py --path tests/expected/ --save 4 --seed 55555 --rescale 8.86
    python deep_continuation/dataset.py --path tests/expected/ --save 4 --seed 555 --rescale 8.86
    python deep_continuation/dataset.py --path tests/expected/ --save 4 --seed 1 --rescale 8.86
    python deep_continuation/dataset.py --path tests/expected/ --save 4 --seed 55555
    python deep_continuation/dataset.py --path tests/expected/ --save 4 --seed 555
    python deep_continuation/dataset.py --path tests/expected/ --save 4 --seed 1
