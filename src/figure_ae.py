from pathlib import Path
from stats import ExperimentResult
import seaborn as sns
import matplotlib.pyplot as plt

files = [
    "results/full_features/newer_college_2020/01_short_experiment/groundtruth_pseudo_planar_point.csv",
    "results/full_features/newer_college_2020/01_short_experiment/constantvelocity_planar.csv",
]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

for f in files:
    exp = ExperimentResult.load(Path(f))
    ae = exp.ae()

    sns.histplot(ae.trans, ax=ax[0], label=exp.params.name)
    sns.histplot(ae.rot, ax=ax[1])

ax[0].legend()
plt.show()
