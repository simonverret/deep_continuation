# DEEP CONTINUATION
© Simon Verret,
Reza Nourafkan,
& Andre-Marie Tremablay

## USAGE
---

Train the default network with:
`$ python deep_continuation.py`

You can pass arguments to this script. For example:
`$ python deep_continuation.py --path ./Database/Training/` 
will fetch the dataset from `./Database/Training/` instead of the default `../data/`. The training and validation data are splitted automatically.

Otherwise, ensure that the training data is a `../data/` directory relative to the current directory. The name of datafiles are:
`../data/SigmaRe.csv`
`../data/Pi.csv`

Another example is of paramter setting:
`$ python deep_continuation.py --lr 0.03` 
will change the learning rate. Refer to the code for more details. All such tunable parameters are also defined in `params.json`.

## RANDOM SEARCH
---
The script `random_search.py` contains further examples of how to automatically change the parameters from a script.

## SLURM
---
The script `submit.sh` show the correct way of using this code on Graham or Cedar (with the dataset and a virtual machine residing on the compute node).

