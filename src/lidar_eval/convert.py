from typing import overload

from evalio.types import SE3 as EvalioSE3
from evalio.types import SO3 as EvalioSO3
from evalio.types import LidarParams as EvalioLidarParams
from loam import LidarParams as LoamLidarParams
from loam import Pose3d as LoamSE3
from loam import Quaterniond as LoamSO3


# fmt: off
# loam -> evalio
@overload
def convert(val: EvalioLidarParams) -> LoamLidarParams: ...
@overload
def convert(val: EvalioSO3) -> LoamSO3: ...
@overload
def convert(val: EvalioSE3) -> LoamSE3: ...

# evalio -> loam
@overload
def convert(val: LoamLidarParams) -> EvalioLidarParams: ...
@overload
def convert(val: LoamSO3) -> EvalioSO3: ...
@overload
def convert(val: LoamSE3) -> EvalioSE3: ...
# fmt: on


def convert(val):
    # From Evalio to Loam
    if isinstance(val, EvalioLidarParams):
        return LoamLidarParams(
            scan_lines=val.num_rows,
            points_per_line=val.num_columns,
            min_range=val.min_range,
            max_range=val.max_range,
        )
    elif isinstance(val, EvalioSO3):
        return LoamSO3(w=val.qw, x=val.qx, y=val.qy, z=val.qz)
    elif isinstance(val, EvalioSE3):
        r = convert(val.rot)
        return LoamSE3(r, val.trans)  # type: ignore

    # From Loam to Evalio
    elif isinstance(val, LoamLidarParams):
        return EvalioLidarParams(
            num_rows=val.scan_lines,
            num_columns=val.points_per_line,
            min_range=val.min_range,
            max_range=val.max_range,
        )
    elif isinstance(val, LoamSO3):
        return EvalioSO3(qw=val.w(), qx=val.x(), qy=val.y(), qz=val.z())
    elif isinstance(val, LoamSE3):
        rot: EvalioSO3 = convert(val.rotation)
        return EvalioSE3(rot, val.translation)

    # bad typeage
    else:
        raise ValueError(f"Invalid type {type(val)} for conversion")
