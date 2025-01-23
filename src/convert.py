from evalio.types import LidarParams as EvalioLidarParams
from loam import LidarParams as LoamLidarParams


def convert(val: EvalioLidarParams) -> LoamLidarParams:
    return LoamLidarParams(
        scan_lines=val.num_rows,
        points_per_line=val.num_columns,
        min_range=val.min_range,
        max_range=val.max_range,
    )
