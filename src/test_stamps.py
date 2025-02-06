from evalio.cli.parser import DatasetBuilder
from evalio.types import LidarMeasurement
import loam

datasets = [
    "helipr/dcc_05",
    "hilti_2022/construction_upper_level_1",
    "multi_campus_2024/ntu_day_01",
    "oxford_spires/observatory_quarter_01",
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
]

for d in datasets:
    print(d)
    data = DatasetBuilder.parse(d)[0].build()

    iterator = iter(data)
    while mm := next(iterator):
        if isinstance(mm, LidarMeasurement):
            break

    print(min(mm.points, key=lambda p: p.t).t)
    print(max(mm.points, key=lambda p: p.t).t)
    print()
