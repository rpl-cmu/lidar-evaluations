from stats import ExperimentResult
from pathlib import Path

er = ExperimentResult.load(
    Path("results/25.02.17_init_all/multi_campus_2024/kth_day_10/ConstantVelocity.csv")
)

er.poses = er.gt[:-1]
er.stamps = er.stamps[1:]
er.gt = er.gt[1:]

print(er.windowed_rte(200).sse().trans)
