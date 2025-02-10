from pathlib import Path
from stats import ExperimentResult
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from wrappers import plt_show
import polars as pl
from dataclasses import asdict
from itertools import product


def moving_average(a, n=40):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


sns.set_style("whitegrid")

dir = Path("results/25.02.06_dewarp_with_init")
datasets = [
    "hilti_2022/construction_upper_level_1",
    "oxford_spires/keble_college_02",
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
]

names = [
    "GroundTruth_GroundTruth.csv",
    "ConstantVelocity_GroundTruth.csv",
    "Identity_GroundTruth.csv",
]

files = [dir / d / n for d, n in product(datasets, names)]

windows = np.arange(0, 1000, 100)
windows[0] = 1


results = []
for f in tqdm(files, leave=False):
    exp = ExperimentResult.load(Path(f))
    exp_dict = asdict(exp.params)
    del exp_dict["features"]
    exp_dict["curvature"] = exp_dict["curvature"].name
    exp_dict["init"] = exp_dict["init"].name
    exp_dict["dewarp"] = exp_dict["dewarp"].name
    for w in tqdm(windows, leave=False):
        rte = exp.windowed_rte(window=int(w)).mean()
        results.append(
            {
                "window": w,
                "rot": rte.rot,
                "trans": rte.trans,
                **exp_dict,
            }
        )

df = pl.DataFrame(results)


fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
sns.lineplot(
    df,
    x="window",
    y="rot",
    ax=ax[0],
    hue="dataset",
    style="dewarp",
    markers=["s", "X", "o"],
    style_order=["Identity", "ConstantVelocity", "GroundTruth"],
    dashes=False,
)
sns.lineplot(
    df,
    x="window",
    y="trans",
    ax=ax[1],
    hue="dataset",
    style="dewarp",
    markers=["s", "X", "o"],
    style_order=["Identity", "ConstantVelocity", "GroundTruth"],
    dashes=False,
    legend=False,
)

ax[0].legend().set_visible(False)
fig.legend(ncol=2, loc="outside lower center")

ax[0].set_title("Rotation")
ax[1].set_title("Translation")

# ax[0].set_yscale("log")
# ax[1].set_yscale("log")

plt_show(Path("figures") / "window_effect.png")
