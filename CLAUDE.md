# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`deep_continuation` is a Python library for analytic continuation of response functions using temperature-agnostic neural networks. The core task is to recover a real-frequency optical conductivity σ(ω) from its imaginary-frequency (Matsubara) representation Π(iωₙ), which is an ill-posed inverse problem in condensed matter physics.

## Install

```bash
pip install -e .
```

## Commands

Run all tests:
```bash
pytest
```

Run a single test file or test function:
```bash
pytest tests/test_dataset.py::test_save
```

Generate dataset from CLI (uses `python-fire` to expose `main()` as a CLI):
```bash
python deep_continuation/dataset.py --save 10000 --seed 555   # validation set
python deep_continuation/dataset.py --save 100000             # training set
python deep_continuation/dataset.py --size 4 --plot           # preview 4 samples
```

Regenerate test expected files after a fundamental change to data generation:
```bash
python tests/test_dataset.py renew
```

## Architecture

The library has three modules:

- **[distribution.py](deep_continuation/distribution.py)** — Random spectral function generators. `get_generator_from_file(param.json, seed)` reads a `param.json` config and returns a generator object (`GaussianMix`, `LorentzMix`, or `BetaMix`). Each generator's `.generate()` method returns a callable `sigma(w)` representing one random conductivity. The Beta variant is the default and most expressive.

- **[conductivity.py](deep_continuation/conductivity.py)** — Physics computations. `sample_on_grid` discretizes σ(ω) onto a uniform frequency grid. `compute_matsubara_response` computes the Matsubara polarization Π(iωₙ) via numerical integration. The key integration routine `integrate_with_tails` uses a three-piece grid (linear center + logarithmic tails) to handle the slow algebraic decay of the integrands.

- **[dataset.py](deep_continuation/dataset.py)** — Dataset generation orchestration. `main()` (exposed as CLI via `fire`) generates and saves `.npy` arrays of `(Pi, sigma, beta, std)` tuples. It skips recomputation when files already exist. Dataset filenames encode all hyperparameters (size, seed, Nwn, Nw, wmax, beta, fixstd).

### Data flow

```
param.json → get_generator_from_file() → distrib.generate() → sigma_func(w)
                                                                     ↓
                                          sample_on_grid()   compute_matsubara_response()
                                                ↓                    ↓
                                           sigma.npy             Pi.npy  ← neural network input
```

### Dataset configuration

Each dataset directory contains a `param.json` specifying the spectral function distribution variant and its hyperparameters (peak number ranges, center/width/weight ranges, Beta `a`/`b` ranges). The `tests/expected/` directory holds pre-generated reference arrays used by regression tests.

### Testing approach

Tests in `test_dataset.py` mock `np.save` and `plot_data` to verify that the generation pipeline produces numerically identical outputs to stored reference arrays (`tests/expected/*.npy`). Tests are parameterized over multiple seeds, `fixstd` values, and `beta` values — so the full suite is large. The `generate_expected()` function regenerates reference files when a deliberate change to data generation logic is made.

### RNG compatibility note

The distribution generators use `np.random.RandomState` (legacy API) for reproducibility with older datasets. Two additional independent RNGs using `np.random.default_rng` handle beta and fixstd sampling. Do not switch the legacy RNG to `default_rng` without regenerating all reference data.
