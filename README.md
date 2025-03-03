# Lidar Odometry Evaluations

This repository is the main source of code for "A Comprehensive Evaluation of Lidar Odometry Techniques". We evaluate a number of core-level LiDAR Odometry (LO) components in a repeatable, reliable manner to draw conclusions about how future LO pipelines should be defined.

Specifically, we test initialization, geometric feature extraction, dewarping, planar feature optimization variants, and feature combination strategies. This is done over a large range of datasets (7 datasets, 47 trajectories) to ensure that the results are generalizable over LiDAR beam sizes, vehicle types, and environments.

## Building

The backbones of these experiments are [evalio](https://github.com/contagon/evalio) for data loading, and a forked version of Dan McGann's MIT-LICENSED [LOAM](https://github.com/contagon/loam), both of which are submodules in the src directory.

As crazy as it sounds, all building is handled through `uv` and `scikit-build-core`. Both evalio and the LOAM implementation will pull their own dependencies. This means to install everything, just running `uv sync` should be enough! 

If you make any changes to LOAM or evalio, it must be signaled to `uv` to rebuild them, 
```bash
touch src/loam/pyproject.toml
uv --verbose sync
```
These are all summarized  in the `justfile` for easy usage.

## Running

All the experiment runners, params, etc can be found in the [src/lidar_eval](src/lidar_eval/) directory, and the experiments ran with all their parameters are in the [experiments](experiments/) directory.

Datasets can be automatically downloaded using evalio,
```bash
uv run evalio download newer_college_2020/*
```
will download all the Newer College 2020 datasets (only Botanic Garden can't be downloaded in this way). A list of all available datasets can be found by running `uv run evalio ls datasets`. 

Next, imu biases must be estimated for the datasets. This can be done by running,
```bash
uv run evalio experiments/generate_imu_bias.py
```

Finally, the experiments can be run with,
```bash
uv run experiments/dewarp.py run -n 10
```
which will run the dewarping experiment with 10 cores.

There is then a `plot` and `stats` subcommand in the dewarp file that can be used to visualize all the results.

General environment parameters can be found in the [env](experiments/env.py), which can tweaked to run a subset of experiments in case of not all datasets being downloaded.

## Contributing
If you notice any bugs or have suggestions, please feel free to open a PR or issue. We are always looking for ways to improve the codebase and the experiments. We'd like this repository to continue to be updated to allow for up to date results to be done as more datasets, bugs, techniques, etc are discovered.