Tests require files in the `expected` directory. Those were create with

    python deep_continuation/dataset.py --path /Users/Simon/codes/deep_continuation_B/tests/expected/ --save 4 --seed 55555 --rescale 8.86
    python deep_continuation/dataset.py --path /Users/Simon/codes/deep_continuation_B/tests/expected/ --save 4 --seed 555 --rescale 8.86
    python deep_continuation/dataset.py --path /Users/Simon/codes/deep_continuation_B/tests/expected/ --save 4 --seed 1 --rescale 8.86
    python deep_continuation/dataset.py --path /Users/Simon/codes/deep_continuation_B/tests/expected/ --save 4 --seed 55555
    python deep_continuation/dataset.py --path /Users/Simon/codes/deep_continuation_B/tests/expected/ --save 4 --seed 555
    python deep_continuation/dataset.py --path /Users/Simon/codes/deep_continuation_B/tests/expected/ --save 4 --seed 1

using a working version of the program