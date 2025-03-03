from evalio.types import LidarMeasurement
from env import SUBSET_TRAJ
from evalio.cli import DatasetBuilder
import numpy as np

all_datasets = DatasetBuilder.parse(SUBSET_TRAJ)

# Save a point cloud
# for d in all_datasets:
#     for mm in d.build():
#         if isinstance(mm, LidarMeasurement):
#             print(d.seq)
#             pts = mm.to_vec_positions()
#             np.save(d.seq + ".npy", pts)
#             break


for d in all_datasets:
    for mm in d.build():
        if isinstance(mm, LidarMeasurement):
            pts = np.asarray(mm.to_vec_positions())
            pts_og = np.load(d.seq + ".npy")
            print(d.seq, np.allclose(pts, pts_og))
            break
