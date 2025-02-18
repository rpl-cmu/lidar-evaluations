from pathlib import Path

RESULTS_DIR = Path("./results")
FIGURE_DIR = Path("./figures")
GRAPHICS_DIR = Path("./graphics")
INC_DATA_DIR = Path("./data")

LEN = 3000

ALL_TRAJ = [
    # ncd20
    "newer_college_2020/01_short_experiment",
    "newer_college_2020/02_long_experiment",
    # ncd21
    "newer_college_2021/quad-easy",
    "newer_college_2021/quad-medium",
    "newer_college_2021/quad-hard",
    "newer_college_2021/stairs",
    "newer_college_2021/cloister",
    "newer_college_2021/maths-easy",
    "newer_college_2021/maths-medium",
    "newer_college_2021/maths-hard",
    # mcd
    "multi_campus_2024/ntu_day_01",
    "multi_campus_2024/ntu_day_02",
    "multi_campus_2024/ntu_day_10",
    "multi_campus_2024/kth_day_06",
    "multi_campus_2024/kth_day_09",
    "multi_campus_2024/kth_day_10",
    "multi_campus_2024/tuhh_day_02",
    "multi_campus_2024/tuhh_day_03",
    "multi_campus_2024/tuhh_day_04",
    # spires
    "oxford_spires/blenheim_palace_01",
    "oxford_spires/blenheim_palace_02",
    "oxford_spires/blenheim_palace_05",
    "oxford_spires/bodleian_library_02",
    "oxford_spires/christ_church_03",
    "oxford_spires/keble_college_02",
    "oxford_spires/keble_college_03",
    "oxford_spires/observatory_quarter_01",
    "oxford_spires/observatory_quarter_02",
    # hilti
    "hilti_2022/construction_upper_level_1",
    "hilti_2022/construction_upper_level_2",
    "hilti_2022/construction_upper_level_3",
    "hilti_2022/basement_2",
    "hilti_2022/attic_to_upper_gallery_2",
    "hilti_2022/corridor_lower_gallery_2",
    # helipr
    "helipr/kaist_05",
    "helipr/kaist_06",
    "helipr/dcc_05",
    "helipr/dcc_06",
    "helipr/riverside_05",
    "helipr/riverside_06",
    # botanic garden
    "botanic_garden/1005_00",
    "botanic_garden/1005_01",
    "botanic_garden/1005_07",
    "botanic_garden/1006_01",
    "botanic_garden/1008_03",
    "botanic_garden/1018_00",
    "botanic_garden/1018_13",
]

# TODO: Verify these are the ones I want
SUBSET_TRAJ = [
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
    "multi_campus_2024/tuhh_day_04",
    "oxford_spires/keble_college_02",
    "hilti_2022/construction_upper_level_1",
    "helipr/kaist_05",
    "botanic_garden/1005_00",
]
