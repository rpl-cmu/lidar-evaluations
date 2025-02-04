from pathlib import Path

import numpy as np

import loam

# ------------------------- params ------------------------- #
lidar_params = loam.LidarParams(
    scan_lines=64, points_per_line=1024, min_range=0.1, max_range=120.0
)
feat_params = loam.FeatureExtractionParams()
feat_params.neighbor_points = 4
feat_params.number_sectors = 6
feat_params.max_edge_feats_per_sector = 10
feat_params.max_planar_feats_per_sector = 50

feat_params.edge_feat_threshold = 50.0
feat_params.planar_feat_threshold = 1.0

feat_params.occlusion_thresh = 0.9
feat_params.parallel_thresh = 0.01

# ------------------------- Load data ------------------------- #
directory = Path("/home/contagon/Research/lidar-features/results/data_dump")
stamps = set(
    (d.stem.split("_")[0], d.stem.split("_")[1]) for d in directory.glob("*.npy")
)
stamps = [
    d
    for d in stamps
    if (directory / f"{d[0]}_{d[1]}_evalio_edge.npy").exists()
    and (directory / f"{d[0]}_{d[1]}_loam_edge.npy").exists()
]

for s in stamps:
    stamp = "_".join(s)

    evalio_edge = np.load(directory / f"{stamp}_evalio_edge.npy")
    loam_edge = np.load(directory / f"{stamp}_loam_edge.npy")

    pts = np.load(directory / f"{stamp}_evalio_pts.npy")
    pts = np.concatenate(
        [pts[i :: lidar_params.scan_lines] for i in range(lidar_params.scan_lines)]
    )
    pts = list(pts)

    # ------------------------- Test ------------------------- #
    print(stamp)
    print("evalio shape:", evalio_edge.shape)
    print("loam shape:", loam_edge.shape)

    if evalio_edge.shape != loam_edge.shape:
        print("shapes do not match")
    else:
        # print("shapes match")
        print("features the same:", np.allclose(evalio_edge, loam_edge))
    print()

# feat_i = loam.extractFeatures(pts, lidar_params, feat_params)
# new_edge = np.asarray(feat_i.edge_points)
# print("this shape: ", new_edge.shape)


# print(
#     "start of evalio and loam match: ",
#     np.allclose(evalio_edge[100:126], loam_edge[100:126]),
# )
# print(loam_edge[123:127])
# print(evalio_edge[123:127])

# try:
#     print("matches with og loam:", np.allclose(new_edge, loam_edge))
# except Exception as _:
#     print("matches with og loam: False")
#     pass

# try:
#     print("matches with evalio: ", np.allclose(new_edge, evalio_edge))
# except Exception as _:
#     print("matches with evalio: False")
