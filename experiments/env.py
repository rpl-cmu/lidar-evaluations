from pathlib import Path
from itertools import chain, combinations
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse

RESULTS_DIR = Path("./results_final")
FIGURE_DIR = Path("./figures")
GRAPHICS_DIR = Path("./graphics")
INC_DATA_DIR = Path("./data")

LEN = 3000

TEXT_WIDTH = 516.0 / 72.27
COL_WIDTH = 252.0 / 72.27

ALL_TRAJ = [
    # ncd20
    "newer_college_2020/short_experiment",
    "newer_college_2020/long_experiment",
    # ncd21
    "newer_college_2021/quad_easy",
    "newer_college_2021/quad_medium",
    "newer_college_2021/quad_hard",
    "newer_college_2021/stairs",
    "newer_college_2021/cloister",
    "newer_college_2021/maths_easy",
    "newer_college_2021/maths_medium",
    "newer_college_2021/maths_hard",
    # mcd
    "multi_campus/ntu_day_01",
    "multi_campus/ntu_day_02",
    "multi_campus/ntu_day_10",
    "multi_campus/kth_day_06",
    "multi_campus/kth_day_09",
    "multi_campus/kth_day_10",
    "multi_campus/tuhh_day_02",
    "multi_campus/tuhh_day_03",
    "multi_campus/tuhh_day_04",
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
    "botanic_garden/b1005_00",
    "botanic_garden/b1005_01",
    "botanic_garden/b1005_07",
    "botanic_garden/b1006_01",
    "botanic_garden/b1008_03",
    "botanic_garden/b1018_00",
    "botanic_garden/b1018_13",
]

SUBSET_TRAJ = [
    "newer_college_2020/short_experiment",
    "newer_college_2021/quad_easy",
    "multi_campus/tuhh_day_04",
    "oxford_spires/keble_college_02",
    "hilti_2022/construction_upper_level_1",
    "helipr/kaist_05",
    "botanic_garden/b1005_00",
]


def plt_show(name: str):
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(GRAPHICS_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    if not is_remote():
        plt.show()


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


def is_remote() -> bool:
    return os.environ.get("SSH_CONNECTION") is not None


def setup_plot():
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    return {
        "Newer College Stereo-Cam": c[0],
        "Newer College Multi-Cam": c[1],
        "Hilti 2022": c[2],
        "Oxford Spires": c[3],
        "Multi-Campus": c[4],
        "HeLiPR": c[5],
        "Botanic Garden": c[6],
    }


def parser(name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{name} experiment")
    subparsers = parser.add_subparsers(dest="action")

    run_parser = subparsers.add_parser("run", help="Run the experiment")
    run_parser.add_argument(
        "-n",
        "--num_threads",
        type=int,
        default=20,
        help="Number of threads to use for multiprocessing",
    )

    stats_parser = subparsers.add_parser("stats", help="Compute statistics")
    stats_parser.add_argument("-s", "--sort", type=str)

    plot_parser = subparsers.add_parser("plot", help="Plot the results")
    plot_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force recomputing results",
    )

    args = parser.parse_args()
    args.name = name
    return args
