from evalio.types import LidarParams as EvalioLidarParams
from loam import LidarParams as LoamLidarParams


def convert(val: EvalioLidarParams) -> LoamLidarParams:
    return LoamLidarParams(val.num_rows, val.num_columns, val.min_range, val.max_range)
