# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/FigureS2_methodology_diagram.py
# Renamed package path: code/figure_drivers/figureS2_methodology_diagram.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

"""Regenerate Supplementary Figure S2 methodology diagram.

This version keeps the figure deterministic and audit-ready: spatial panels
come from real project-rendered rasters, while the workflow, process-neighborhood
tables, and ALE glyphs are drawn as publication graphics with no numeric
thresholds.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RASTER = ROOT / "make_diagram" / "outputs" / "rendered_rasters"
REAL = ROOT / "pnas_nexus_submission" / "results" / "cache" / "figure_s2_refs" / "real_asset_revision"
FIG_DIR = ROOT / "pnas_nexus_submission" / "results" / "figures"

PNG_OUT = FIG_DIR / "FigureS2_methodology_diagram.png"
PDF_OUT = FIG_DIR / "FigureS2_methodology_diagram.pdf"

W, H = 7200, 3960

PF = "#5A8F63"
NPF = "#9A6A49"
DU = "#1E5BAA"
GRAD = "#C1272D"
TL = "#2F7F8A"
RTS = "#C23B6E"
GRAY = "#5F6670"
LIGHT_GRAY = "#D9DEE5"

BLUE_EDGE = "#79AEEA"
GREEN_EDGE = "#66A56E"
YELLOW_EDGE = "#E0A32B"
ORANGE_EDGE = "#E78549"
PURPLE_EDGE = "#A68BE8"
FONT_SCALE = 0.58


def setup() -> tuple[plt.Figure, plt.Axes]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(12.0, 6.6), dpi=600, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str = "white",
    ec: str = LIGHT_GRAY,
    lw: float = 1.0,
    radius: float = 20,
    alpha: float = 1.0,
    z: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=patches.BoxStyle("Round", pad=0.018, rounding_size=radius),
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    s: str,
    *,
    size: float = 8.0,
    color: str = "#1F252D",
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    rotation: float = 0,
    linespacing: float = 1.12,
    z: int = 10,
) -> None:
    ax.text(
        x,
        y,
        s,
        fontsize=size * FONT_SCALE,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        linespacing=linespacing,
        zorder=z,
    )


def panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    fc: str,
    ec: str,
    title_color: str,
) -> None:
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=2.0, radius=20, alpha=0.74, z=0)
    label(ax, x + w / 2, y + 72, title, size=17, color=title_color, weight="bold")


def arrow(
    ax: plt.Axes,
    xy1: tuple[float, float],
    xy2: tuple[float, float],
    *,
    lw: float = 1.2,
    color: str = GRAY,
    dashed: bool = False,
    ms: float = 16,
    rad: float = 0.0,
    z: int = 7,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=(0, (5, 4)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=4,
            zorder=z,
        )
    )


def read_image(path: Path, *, crop: bool = True) -> np.ndarray:
    img = mpimg.imread(path)
    if not crop:
        return img
    rgb = img[..., :3]
    if img.shape[-1] == 4:
        mask = img[..., 3] > 0.02
    else:
        mask = np.mean(rgb, axis=-1) < 0.985
    if not np.any(mask):
        return img
    yy, xx = np.where(mask)
    pad = 10
    y0 = max(0, yy.min() - pad)
    y1 = min(img.shape[0], yy.max() + pad)
    x0 = max(0, xx.min() - pad)
    x1 = min(img.shape[1], xx.max() + pad)
    return img[y0:y1, x0:x1, :]


def place_image(
    ax: plt.Axes,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    crop: bool = True,
    frame: bool = False,
    ec: str = LIGHT_GRAY,
    lw: float = 0.7,
    z: int = 4,
) -> tuple[float, float, float, float]:
    img = read_image(path, crop=crop)
    ih, iw = img.shape[:2]
    box_ratio = w / h
    img_ratio = iw / ih
    if img_ratio > box_ratio:
        dw = w
        dh = w / img_ratio
    else:
        dh = h
        dw = h * img_ratio
    px = x + (w - dw) / 2
    py = y + (h - dh) / 2
    ax.imshow(img, extent=(px, px + dw, py + dh, py), zorder=z)
    if frame:
        rounded(ax, x, y, w, h, fc="none", ec=ec, lw=lw, radius=8, z=z + 1)
    return px, py, dw, dh


def simple_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    txt: str,
    *,
    fc: str = "white",
    ec: str = "#AEB8C4",
    color: str = "#222",
    lw: float = 1.0,
    size: float = 9,
    weight: str = "normal",
    radius: float = 10,
) -> None:
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, radius=radius, z=3)
    label(ax, x + w / 2, y + h / 2, txt, size=size, color=color, weight=weight)


def image_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    path: Path,
    title: str,
    *,
    color: str = "#222",
    crop: bool = True,
    ec: str = LIGHT_GRAY,
    title_size: float = 9.0,
    image_pad: float = 12,
) -> None:
    rounded(ax, x, y, w, h, fc="white", ec=ec, lw=0.9, radius=10, z=2)
    if title:
        label(ax, x + w / 2, y + 28, title, size=title_size, color=color, weight="bold")
        title_extra = 22 * max(0, title.count("\n"))
        img_y = y + 52 + title_extra
        img_h = h - 62 - title_extra
    else:
        img_y = y + image_pad
        img_h = h - 2 * image_pad
    place_image(ax, path, x + image_pad, img_y, w - 2 * image_pad, img_h, crop=crop, z=3)


def source_row(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    title: str,
    source: str,
    path: Path,
    *,
    img_crop: bool = True,
) -> None:
    source_y = y + 104 + 54 * max(0, title.count("\n"))
    label(ax, x + 32, y + 44, title, size=9.2, weight="bold", ha="left", va="top")
    label(
        ax,
        x + 32,
        source_y,
        source,
        size=7.4,
        color="#34465A",
        ha="left",
        va="top",
        linespacing=1.16,
    )
    place_image(ax, path, x + 412, y + 18, w - 450, 260, crop=img_crop, frame=True, z=4)


def data_sources(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    panel(ax, x, y, w, h, "Data sources", fc="#EEF5FF", ec=BLUE_EDGE, title_color=DU)
    rows = [
        ("Sentinel-1 SAR", "Copernicus S1 SAR\nSLC stack", RASTER / "sentinel1_cube_stack.png"),
        ("DEM / terrain", "Copernicus GLO-30\nterrain context", RASTER / "dem.png"),
        ("Environmental rasters", "Fig. S1 covariate\nstack", RASTER / "environment_stack_key8.png"),
        ("PF / NPF map", "Zou et al. TTOP\nPF/NPF map", RASTER / "permafrost_distribution.png"),
        ("Railway / stations", "Railway and\nstation sites", REAL / "railway_hillshade_no_coords.png"),
        ("Lake/slump\ninventories", "Thermokarst lakes and\nthaw slumps", REAL / "abrupt_thaw_inventory_real_no_axes.png"),
    ]
    yy = y + 150
    for idx, (title, source, path) in enumerate(rows):
        source_row(ax, x + 20, yy + idx * 380, w - 40, title, source, path)
        if idx < len(rows) - 1:
            ax.plot([x + 35, x + w - 35], [yy + idx * 380 + 315, yy + idx * 380 + 315], color="#D8E2EF", lw=0.8, zorder=3)


def insar_processing(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    panel(ax, x, y, w, h, "Deriving deformation data", fc="#F0F8ED", ec=GREEN_EDGE, title_color="#1E692E")
    cx = x + w / 2
    simple_box(ax, x + 310, y + 170, 600, 78, "ISCE2 preprocessing", fc="#FFFFFF", ec=GREEN_EDGE, size=9.5)
    simple_box(ax, x + 310, y + 305, 600, 88, "MintPy SBAS TS-InSAR", fc="#FFFFFF", ec=GREEN_EDGE, size=9.5)
    simple_box(ax, x + 95, y + 495, 345, 92, "Ascending LOS", fc="#F5FBFC", ec="#67AEBB", size=8.8)
    simple_box(ax, x + w - 440, y + 495, 345, 92, "Descending LOS", fc="#F5FBFC", ec="#67AEBB", size=8.8)
    diamond = patches.RegularPolygon((cx, y + 700), numVertices=4, radius=118, orientation=0, fc="#FFFFFF", ec=GREEN_EDGE, lw=1.0, zorder=3)
    ax.add_patch(diamond)
    label(ax, cx, y + 700, "Both LOS\navailable?", size=7.8, color="#244B2C", linespacing=1.0)
    simple_box(ax, x + 330, y + 850, 560, 86, "2D decomposition", fc="#FFFFFF", ec=GREEN_EDGE, size=9.8)

    du_x, du_y, card_w, card_h = x + 85, y + 1135, w - 170, 505
    rounded(ax, du_x, du_y, card_w, card_h, fc="white", ec=DU, lw=1.6, radius=12, z=2)
    label(ax, du_x + 45, du_y + 58, r"$d_u$", size=14.5, color=DU, weight="bold", ha="left")
    label(ax, du_x + 180, du_y + 58, "vertical velocity", size=9.5, ha="left")
    place_image(ax, RASTER / "d_u.png", du_x + 320, du_y + 105, card_w - 360, card_h - 132, crop=True, z=4)

    grad_x, grad_y = x + 85, y + 1900
    rounded(ax, grad_x, grad_y, card_w, card_h, fc="white", ec=GRAD, lw=1.6, radius=12, z=2)
    label(ax, grad_x + 45, grad_y + 58, r"$|\nabla d_u|$", size=14.5, color=GRAD, weight="bold", ha="left")
    label(ax, grad_x + 250, grad_y + 58, "gradient magnitude", size=9.5, ha="left")
    place_image(ax, RASTER / "grad_mag_km.png", grad_x + 320, grad_y + 105, card_w - 360, card_h - 132, crop=True, z=4)

    derive_x, derive_y, derive_w, derive_h = cx - 180, y + 1715, 360, 78
    simple_box(ax, derive_x, derive_y, derive_w, derive_h, "Derive gradient", fc="#FFFFFF", ec="#AAB1BA", size=8.8)
    arrow(ax, (cx, y + 248), (cx, y + 302), ms=12)
    arrow(ax, (cx, y + 393), (x + 268, y + 492), ms=12)
    arrow(ax, (cx, y + 393), (x + w - 268, y + 492), ms=12)
    arrow(ax, (x + 268, y + 587), (cx - 75, y + 645), ms=12)
    arrow(ax, (x + w - 268, y + 587), (cx + 75, y + 645), ms=12)
    arrow(ax, (cx, y + 820), (cx, y + 846), ms=12)
    label(ax, cx + 70, y + 828, "Yes", size=7.8, ha="left")
    arrow(ax, (cx, y + 936), (cx, du_y - 4), ms=12)
    arrow(ax, (cx, du_y + card_h + 10), (cx, derive_y - 8), color=DU, ms=10)
    arrow(ax, (cx, derive_y + derive_h + 10), (cx, grad_y - 12), color=GRAD, ms=10)


def distance_registration(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    panel(ax, x, y, w, h, "Distance registration", fc="#FFF8E8", ec=YELLOW_EDGE, title_color="#9A5C00")

    def distance_chip(
        yy: float,
        title: str,
        path: Path,
        title_color: str,
        edge_color: str,
        boxes: tuple[str, str],
        note: str,
    ) -> None:
        rounded(ax, x + 45, yy, w - 90, 1030, fc="white", ec=edge_color, lw=1.2, radius=12, z=1)
        label(ax, x + 70, yy + 46, title, size=10.5, color=title_color, weight="bold", ha="left")
        place_image(ax, path, x + 85, yy + 105, 540, 790, crop=True, z=3)
        arrow(ax, (x + 650, yy + 500), (x + 750, yy + 400), ms=13)
        arrow(ax, (x + 650, yy + 500), (x + 750, yy + 642), ms=13)
        simple_box(ax, x + 770, yy + 300, 380, 130, boxes[0], fc="#FFFFFF", ec=edge_color, color=title_color, size=10.0, weight="bold")
        simple_box(ax, x + 770, yy + 545, 380, 130, boxes[1], fc="#FFFFFF", ec=edge_color, color=title_color, size=10.0, weight="bold")
        label(ax, x + w / 2, yy + 960, note, size=8.0, color="#6B4B13")

    distance_chip(
        y + 150,
        r"$d_B$  boundary distance",
        REAL / "dB_boundary_distance_real_no_axes.png",
        DU,
        "#6AA4FF",
        ("Transition", "Interior"),
        "neighborhood grouping only",
    )
    distance_chip(
        y + 1325,
        r"$d_A$ lake/slump distance",
        REAL / "dA_abrupt_thaw_distance_real_no_axes.png",
        GRAD,
        "#FF6262",
        ("Lake/slump\nneighborhood", "Background"),
        "neighborhood grouping only",
    )


def conditioned_extremes(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    panel(ax, x, y, w, h, "Conditioned extremes", fc="#FFF4EC", ec=ORANGE_EDGE, title_color="#9E4B16")
    card_x = x + 55
    card_w = 1055
    image_card(
        ax,
        card_x,
        y + 170,
        card_w,
        800,
        REAL / "conditioned_du_distribution_fig02_compact_no_numbers.png",
        "",
        crop=False,
        ec="#E8C7B6",
        image_pad=10,
    )
    image_card(
        ax,
        card_x,
        y + 1055,
        card_w,
        800,
        REAL / "conditioned_grad_distribution_fig02_compact_no_numbers.png",
        "",
        crop=False,
        ec="#E8C7B6",
        image_pad=10,
    )

    bx = x + 1160
    rounded(ax, bx - 24, y + 220, 360, 680, fc="none", ec=PF, lw=2.0, radius=12, z=5)
    label(ax, bx + 155, y + 270, "PF targets", size=9.5, color=PF, weight="bold")
    simple_box(ax, bx, y + 360, 315, 122, r"PF $d_u$", fc="#FFFFFF", ec=PF, color=DU, size=10.3, weight="bold")
    simple_box(ax, bx, y + 585, 315, 122, r"PF $|\nabla d_u|$", fc="#FFFFFF", ec=PF, color=GRAD, size=10.3, weight="bold")
    rounded(ax, bx - 24, y + 1125, 360, 680, fc="none", ec=NPF, lw=2.0, radius=12, z=5)
    label(ax, bx + 155, y + 1175, "NPF targets", size=9.5, color=NPF, weight="bold")
    simple_box(ax, bx, y + 1265, 315, 122, r"NPF $d_u$", fc="#FFFFFF", ec=NPF, color=DU, size=10.3, weight="bold")
    simple_box(ax, bx, y + 1490, 315, 122, r"NPF $|\nabla d_u|$", fc="#FFFFFF", ec=NPF, color=GRAD, size=10.3, weight="bold")
    label(ax, x + w / 2, y + 2090, "domain-specific extreme signals", size=11.0, color="#7A3D0B", weight="bold")


def contrast_glyph(ax: plt.Axes, x: float, y: float) -> None:
    for row, yy in enumerate([0, 92, 184]):
        for col in range(4):
            rounded(
                ax,
                x + col * 76,
                y + yy,
                58,
                52,
                fc=["#EEE8FA", "#EFF8EE", "#FFF0E6"][row],
                ec="#9277C0",
                lw=0.8,
                radius=5,
                z=4,
            )
    ax.plot([x + 110, x + 170], [y + 258, y + 350], color="#777", lw=1.0, zorder=3)
    ax.plot([x + 230, x + 180], [y + 258, y + 350], color="#777", lw=1.0, zorder=3)


def tree_icon(ax: plt.Axes, x: float, y: float, *, scale: float = 1.0, color: str = "#6E5AA8") -> None:
    pts = np.array(
        [
            [0, 0],
            [-22, 34],
            [22, 34],
            [-36, 70],
            [-10, 70],
            [10, 70],
            [36, 70],
        ],
        dtype=float,
    )
    pts *= scale
    pts[:, 0] += x
    pts[:, 1] += y
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for a, b in edges:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color, lw=0.9, zorder=5)
    for idx, (px, py) in enumerate(pts):
        radius = 7.5 * scale if idx == 0 else 6.5 * scale
        ax.add_patch(patches.Circle((px, py), radius=radius, fc="white", ec=color, lw=0.85, zorder=6))


def learner_box(ax: plt.Axes, x: float, y: float, w: float, h: float, txt: str) -> None:
    rounded(ax, x, y, w, h, fc="#FBFAFF", ec="#A890D8", lw=0.8, radius=8, z=3)
    tree_icon(ax, x + w / 2, y + 24, scale=0.72)
    label(ax, x + w / 2, y + h - 22, txt, size=7.5, color="#352179", weight="bold")


def stack_model_glyph(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded(ax, x, y, w, h, fc="#FFFFFF", ec="#7D61B8", lw=1.1, radius=12, z=2)
    label(
        ax,
        x + w / 2,
        y + 48,
        "Domain-specific\nno-PCA ExtraTrees diagnostics",
        size=8.9,
        color="#352179",
        weight="bold",
        linespacing=0.95,
    )
    label(ax, x + w / 2, y + 105, "median imputation; original predictor variables", size=7.3, color="#6B5C8E")
    model_w, model_h = 320, 165
    model_x = x + (w - model_w) / 2
    model_y = y + 145
    learner_box(ax, model_x, model_y, model_w, model_h, "ExtraTrees")
    out_w, out_h = 245, 92
    left_x = x + 125
    right_x = x + w - 125 - out_w
    out_y = y + 405
    for dest_x in (left_x, right_x):
        ax.plot(
            [model_x + model_w / 2, dest_x + out_w / 2],
            [model_y + model_h + 12, out_y - 10],
            color="#6E5AA8",
            lw=0.8,
            alpha=0.9,
            zorder=4,
        )
    simple_box(
        ax,
        left_x,
        out_y,
        out_w,
        out_h,
        "Component\ngrouped PI",
        fc="#F7F3FF",
        ec="#7D61B8",
        color="#352179",
        size=8.8,
        weight="bold",
    )
    simple_box(
        ax,
        right_x,
        out_y,
        out_w,
        out_h,
        "Neighborhood\nALE refits",
        fc="#F7F3FF",
        ec="#7D61B8",
        color="#352179",
        size=8.8,
        weight="bold",
    )
    label(ax, x + w / 2, y + h - 28, "extreme signal susceptibility", size=7.6, color="#352179")


def susceptibility_panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    panel(ax, x, y, w, h, "Extreme signal susceptibility", fc="#F5F0FF", ec=PURPLE_EDGE, title_color="#3E278E")
    image_card(ax, x + 70, y + 155, 360, 510, RASTER / "environment_stack_key8.png", "Environmental\npredictors", crop=True, ec="#C8B7EF", title_size=9.2)
    rounded(ax, x + 500, y + 155, 390, 510, fc="white", ec="#C8B7EF", lw=0.9, radius=10, z=2)
    label(ax, x + 695, y + 210, "Environmental\nspatial contrasts", size=9.0)
    contrast_glyph(ax, x + 555, y + 290)
    stack_model_glyph(ax, x + 990, y + 135, 825, 585)
    arrow(ax, (x + 430, y + 700), (x + 982, y + 675), ms=13, rad=-0.08)
    arrow(ax, (x + 895, y + 420), (x + 982, y + 420), ms=13)
    arrow(ax, (x + 1402, y + 720), (x + 1402, y + 885), ms=15)

    label(ax, x + w / 2, y + 890, "domain-target susceptibility signals", size=11.2, color="#352179", weight="bold")
    group_y, group_h = y + 960, 1340
    group_w = 820
    rounded(ax, x + 80, group_y, group_w, group_h, fc="#FFFFFF", ec=PF, lw=2.0, radius=12, z=1)
    rounded(ax, x + 1025, group_y, group_w, group_h, fc="#FFFFFF", ec=NPF, lw=2.0, radius=12, z=1)
    label(ax, x + 490, group_y + 65, "PF", size=12, color=PF, weight="bold")
    label(ax, x + 1435, group_y + 65, "NPF", size=12, color=NPF, weight="bold")
    image_card(ax, x + 125, group_y + 145, 340, 610, REAL / "susceptibility_pf_d_u_real_no_axes.png", r"$d_u$", color=DU, crop=True, ec="#E5E9F2", title_size=9.5)
    image_card(ax, x + 520, group_y + 145, 340, 610, REAL / "susceptibility_pf_grad_mag_km_real_no_axes.png", r"$|\nabla d_u|$", color=GRAD, crop=True, ec="#E5E9F2", title_size=9.5)
    image_card(ax, x + 1070, group_y + 145, 340, 610, REAL / "susceptibility_npf_d_u_real_no_axes.png", r"$d_u$", color=DU, crop=True, ec="#E5E9F2", title_size=9.5)
    image_card(ax, x + 1465, group_y + 145, 340, 610, REAL / "susceptibility_npf_grad_mag_km_real_no_axes.png", r"$|\nabla d_u|$", color=GRAD, crop=True, ec="#E5E9F2", title_size=9.5)
    simple_box(ax, x + 125, group_y + 820, 340, 88, r"PF $d_u$", fc="#FFFFFF", ec=PF, color=DU, size=10.0, weight="bold")
    simple_box(ax, x + 520, group_y + 820, 340, 88, r"PF $|\nabla d_u|$", fc="#FFFFFF", ec=PF, color=GRAD, size=10.0, weight="bold")
    simple_box(ax, x + 1070, group_y + 820, 340, 88, r"NPF $d_u$", fc="#FFFFFF", ec=NPF, color=DU, size=10.0, weight="bold")
    simple_box(ax, x + 1465, group_y + 820, 340, 88, r"NPF $|\nabla d_u|$", fc="#FFFFFF", ec=NPF, color=GRAD, size=10.0, weight="bold")
    simple_box(ax, x + 125, group_y + 1030, 340, 145, "Trained PF\n" + r"$d_u$" + "\npredictor", fc="#FFFFFF", ec=PF, color=DU, size=8.2, weight="bold")
    simple_box(ax, x + 520, group_y + 1030, 340, 145, "Trained PF\n" + r"$|\nabla d_u|$" + "\npredictor", fc="#FFFFFF", ec=PF, color=GRAD, size=8.2, weight="bold")
    simple_box(ax, x + 1070, group_y + 1030, 340, 145, "Trained NPF\n" + r"$d_u$" + "\npredictor", fc="#FFFFFF", ec=NPF, color=DU, size=8.2, weight="bold")
    simple_box(ax, x + 1465, group_y + 1030, 340, 145, "Trained NPF\n" + r"$|\nabla d_u|$" + "\npredictor", fc="#FFFFFF", ec=NPF, color=GRAD, size=8.2, weight="bold")


def mini_profile(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, mode: str) -> None:
    rounded(ax, x, y, w, h, fc="#FFFFFF", ec="#D1D7E0", lw=0.8, radius=8, z=2)
    label(ax, x + 24, y + 24, title, size=8.4, color="#332B73", weight="bold", ha="left")
    left = x + 40
    right = x + w - 36
    top = y + 70
    bottom = y + h - 55
    mid = left + 0.52 * (right - left)
    if mode == "db":
        ax.add_patch(patches.Rectangle((left, top), mid - left, bottom - top, fc=NPF, ec="none", alpha=0.13, zorder=2))
        ax.add_patch(patches.Rectangle((mid, top), right - mid, bottom - top, fc=PF, ec="none", alpha=0.13, zorder=2))
        ax.plot([mid, mid], [top, bottom], color="#5A5A5A", lw=0.9, ls=(0, (4, 3)), zorder=3)
        label(ax, left + 0.25 * (right - left), top + 24, "Transition", size=7.4, color=NPF)
        label(ax, right - 0.20 * (right - left), top + 24, "Interior", size=7.4, color=PF)
        t = np.linspace(0, 1, 100)
        blue = 0.25 + 0.08 * np.sin(2.2 * np.pi * t) + 0.52 / (1 + np.exp((t - 0.56) * 18))
        red = 0.78 - 0.58 / (1 + np.exp((t - 0.56) * 18)) + 0.04 * np.sin(2 * np.pi * t)
    else:
        ax.add_patch(patches.Rectangle((left, top), 0.58 * (right - left), bottom - top, fc=RTS, ec="none", alpha=0.10, zorder=2))
        ax.add_patch(patches.Rectangle((left + 0.58 * (right - left), top), 0.42 * (right - left), bottom - top, fc=PF, ec="none", alpha=0.10, zorder=2))
        xcut = left + 0.58 * (right - left)
        ax.plot([xcut, xcut], [top, bottom], color=GRAD, lw=0.9, ls=(0, (4, 3)), zorder=3)
        label(ax, left + 0.30 * (right - left), top + 24, "Impacted", size=7.4, color=GRAD)
        label(ax, right - 0.20 * (right - left), top + 24, "Background", size=7.4, color=PF)
        t = np.linspace(0, 1, 100)
        blue = 0.55 - 0.16 * t + 0.06 * np.sin(2.4 * np.pi * t)
        red = 0.72 - 0.28 * t + 0.10 * np.sin(2.1 * np.pi * t)
    xs = left + t * (right - left)
    ax.plot(xs, top + blue * (bottom - top), color=DU, lw=2.0, zorder=5)
    ax.plot(xs, top + red * (bottom - top), color=GRAD, lw=2.0, zorder=5)
    label(ax, x + w / 2, y + h - 20, r"blue: $d_u$    red: $|\nabla d_u|$", size=7.2, color="#332B73")


def process_tables(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded(ax, x, y, w, h, fc="white", ec="#C8B7EF", lw=0.9, radius=10, z=1)
    label(ax, x + 25, y + 34, "Fig. 4C/D", size=9.6, color="#332B73", weight="bold", ha="left")
    label(ax, x + 25, y + 77, "neighborhood tables", size=8.4, color="#332B73", ha="left")

    label(ax, x + 385, y + 125, "NPF: two zones", size=9.5, color=NPF, weight="bold")
    n_x, n_y, n_w, n_h = x + 70, y + 175, 620, 205
    simple_box(ax, n_x, n_y, n_w, n_h, "Transition", fc="#FBF7F2", ec=NPF, color=NPF, size=11.2, weight="bold")
    simple_box(ax, n_x, n_y + 250, n_w, n_h, "Interior", fc="#FBF7F2", ec=NPF, color=NPF, size=11.2, weight="bold")

    pf_x = x + 780
    label(ax, pf_x + 555, y + 125, "PF: four zones", size=9.5, color=PF, weight="bold")
    headers = [("Lake/slump\nneighborhood", GRAD), ("Background", "#424242")]
    for i, (txt, col) in enumerate(headers):
        label(ax, pf_x + 425 + i * 530, y + 210, txt, size=7.8, color=col, weight="bold")
    row_labels = [("Transition", y + 290), ("Interior", y + 545)]
    for row_txt, yy in row_labels:
        label(ax, pf_x + 170, yy + 70, row_txt, size=8.8, color=PF, weight="bold", ha="right")
    specs = [
        ("Transition +\nlake/slump", GRAD, pf_x + 225, y + 275),
        ("Transition\nbackground", PF, pf_x + 755, y + 275),
        ("Interior +\nlake/slump", GRAD, pf_x + 225, y + 530),
        ("Interior\nbackground", PF, pf_x + 755, y + 530),
    ]
    for txt, col, xx, yy in specs:
        simple_box(ax, xx, yy, 430, 190, txt, fc="#FFF8F8" if col == GRAD else "#F8FCF8", ec=PF, color=col, size=10.0, weight="bold")


def draw_curve(ax: plt.Axes, x: float, y: float, w: float, h: float, t: np.ndarray, vals: np.ndarray, *, color: str, ls: object, lw: float = 1.15) -> None:
    left = x + 34
    right = x + w - 28
    top = y + 86
    bottom = y + h - 52
    xs = left + t * (right - left)
    ys = bottom - vals * (bottom - top)
    ax.fill_between(xs, ys - 13, ys + 13, color=color, alpha=0.040, linewidth=0, zorder=4)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", dash_capstyle="round", zorder=5)


def curve_from_knots(t: np.ndarray, knots: list[tuple[float, float]]) -> np.ndarray:
    xs, ys = zip(*knots)
    return np.interp(t, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))


def mini_ale(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, *, domain: str, feature: str) -> None:
    tint = NPF if domain == "npf" else PF
    rounded(ax, x, y, w, h, fc="#FFFFFF", ec="#C9CDD6", lw=0.8, radius=9, z=2)
    ax.add_patch(patches.Rectangle((x + 22, y + 70), w - 44, h - 115, fc=tint, ec="none", alpha=0.075, zorder=2))
    label(ax, x + w / 2, y + 38, title, size=9.2, color=tint, weight="bold")
    zero_y = y + 0.56 * h
    ax.plot([x + 34, x + w - 28], [zero_y, zero_y], color="#AAB1BA", lw=0.65, ls=(0, (4, 3)), zorder=3)
    t = np.linspace(0, 1, 160)
    if feature == "magt":
        styles = [((0, (7, 4)), "Interior"), ((0, (5, 2, 1.2, 2)), "Transition")]
        blue_patterns = [
            [(0.00, 0.84), (0.22, 0.82), (0.38, 0.76), (0.55, 0.66), (0.72, 0.62), (1.00, 0.61)],
            [(0.00, 0.76), (0.24, 0.81), (0.36, 0.77), (0.52, 0.63), (0.70, 0.57), (1.00, 0.55)],
        ]
        red_patterns = [
            [(0.00, 0.41), (0.22, 0.44), (0.40, 0.39), (0.62, 0.28), (0.82, 0.21), (1.00, 0.22)],
            [(0.00, 0.35), (0.24, 0.38), (0.38, 0.33), (0.55, 0.23), (0.75, 0.18), (1.00, 0.17)],
        ]
    elif feature == "twi":
        styles = [((0, (7, 4)), ""), ((0, (5, 2, 1.2, 2)), ""), ((0, (1.2, 2.2)), ""), ((0, (7, 2, 1.2, 2, 1.2, 2)), "")]
        blue_patterns = [
            [(0.00, 0.62), (0.18, 0.63), (0.34, 0.78), (0.43, 0.80), (0.58, 0.63), (0.78, 0.54), (1.00, 0.45)],
            [(0.00, 0.60), (0.18, 0.61), (0.32, 0.76), (0.44, 0.72), (0.58, 0.57), (0.78, 0.48), (1.00, 0.40)],
            [(0.00, 0.67), (0.18, 0.70), (0.34, 0.82), (0.44, 0.66), (0.58, 0.50), (0.78, 0.42), (1.00, 0.34)],
            [(0.00, 0.69), (0.18, 0.72), (0.34, 0.80), (0.45, 0.70), (0.62, 0.64), (0.82, 0.55), (1.00, 0.49)],
        ]
        red_patterns = [
            [(0.00, 0.45), (0.18, 0.42), (0.34, 0.34), (0.50, 0.28), (0.72, 0.22), (1.00, 0.15)],
            [(0.00, 0.34), (0.18, 0.41), (0.32, 0.50), (0.48, 0.44), (0.68, 0.36), (1.00, 0.25)],
            [(0.00, 0.39), (0.18, 0.43), (0.34, 0.40), (0.50, 0.35), (0.72, 0.30), (1.00, 0.22)],
            [(0.00, 0.35), (0.18, 0.45), (0.32, 0.52), (0.48, 0.49), (0.68, 0.45), (1.00, 0.37)],
        ]
    else:
        styles = [((0, (7, 4)), ""), ((0, (5, 2, 1.2, 2)), ""), ((0, (1.2, 2.2)), ""), ((0, (7, 2, 1.2, 2, 1.2, 2)), "")]
        blue_patterns = [
            [(0.00, 0.59), (0.20, 0.60), (0.34, 0.66), (0.48, 0.78), (0.66, 0.82), (0.84, 0.79), (1.00, 0.74)],
            [(0.00, 0.56), (0.20, 0.57), (0.34, 0.62), (0.48, 0.70), (0.66, 0.74), (0.84, 0.73), (1.00, 0.68)],
            [(0.00, 0.61), (0.20, 0.62), (0.34, 0.64), (0.50, 0.66), (0.70, 0.66), (0.86, 0.64), (1.00, 0.61)],
            [(0.00, 0.55), (0.20, 0.56), (0.34, 0.62), (0.50, 0.75), (0.68, 0.85), (0.86, 0.84), (1.00, 0.82)],
        ]
        red_patterns = [
            [(0.00, 0.32), (0.18, 0.30), (0.34, 0.35), (0.52, 0.42), (0.70, 0.45), (0.86, 0.43), (1.00, 0.34)],
            [(0.00, 0.36), (0.18, 0.39), (0.34, 0.37), (0.50, 0.33), (0.70, 0.30), (1.00, 0.27)],
            [(0.00, 0.33), (0.18, 0.34), (0.34, 0.31), (0.52, 0.29), (0.72, 0.31), (1.00, 0.30)],
            [(0.00, 0.39), (0.18, 0.34), (0.34, 0.30), (0.52, 0.38), (0.72, 0.42), (1.00, 0.45)],
        ]

    for idx, (ls, _name) in enumerate(styles):
        blue_vals = curve_from_knots(t, blue_patterns[idx])
        red_vals = curve_from_knots(t, red_patterns[idx])
        draw_curve(ax, x, y, w, h, t, np.clip(blue_vals, 0.50, 0.91), color=DU, ls=ls, lw=1.0)
        draw_curve(ax, x, y, w, h, t, np.clip(red_vals, 0.12, 0.52), color=GRAD, ls=ls, lw=0.98)


def line_key(ax: plt.Axes, x: float, y: float, label_txt: str, ls: object, color: str = "#333", size: float = 7.8) -> None:
    ax.plot([x, x + 78], [y, y], color=color, lw=2.1, ls=ls, solid_capstyle="round", dash_capstyle="round", zorder=8)
    label(ax, x + 96, y, label_txt, size=size, color="#333", ha="left")


def ale_panel_lower(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded(ax, x, y, w, h, fc="white", ec="#C8B7EF", lw=0.9, radius=10, z=1)
    label(ax, x + 25, y + 34, "Fig. 5", size=9.6, color="#332B73", weight="bold", ha="left")
    label(ax, x + 25, y + 77, "ALE refits by neighborhood", size=8.4, color="#332B73", ha="left")
    panel_w = (w - 155) / 3
    mini_ale(ax, x + 45, y + 135, panel_w, 430, "NPF · MAGT", domain="npf", feature="magt")
    mini_ale(ax, x + 75 + panel_w, y + 135, panel_w, 430, "PF · TWI", domain="pf", feature="twi")
    mini_ale(ax, x + 105 + 2 * panel_w, y + 135, panel_w, 430, "PF · NDVI", domain="pf", feature="ndvi")

    key_y = y + 650
    label(ax, x + 55, key_y - 42, "NPF zones", size=8.3, color=NPF, weight="bold", ha="left")
    line_key(ax, x + 55, key_y, "Transition", (0, (5, 2, 1.2, 2)))
    line_key(ax, x + 55, key_y + 60, "Interior", (0, (7, 4)))

    label(ax, x + 580, key_y - 42, "PF zones", size=8.3, color=PF, weight="bold", ha="left")
    line_key(ax, x + 580, key_y, "Transition + lake/slump", (0, (7, 4)), size=7.5)
    line_key(ax, x + 580, key_y + 60, "Transition background", (0, (5, 2, 1.2, 2)), size=7.5)
    line_key(ax, x + 1115, key_y, "Interior + lake/slump", (0, (1.2, 2.2)), size=7.5)
    line_key(ax, x + 1115, key_y + 60, "Interior background", (0, (7, 2, 1.2, 2, 1.2, 2)), size=7.5)
    label(ax, x + w / 2, y + h - 42, r"blue: $d_u$ susceptibility     red: $|\nabla d_u|$ susceptibility", size=8.4, color="#332B73", weight="bold")


def lower_band(ax: plt.Axes) -> None:
    x, y, w, h = 145, 2705, 6910, 1085
    rounded(ax, x, y, w, h, fc="#FAFAFF", ec=PURPLE_EDGE, lw=1.8, radius=18, z=0)
    label(ax, x + w / 2, y + 70, "ALE / neighborhood inspection", size=22, color="#332B73", weight="bold")

    c1_x, c1_w = x + 65, 1460
    rounded(ax, c1_x, y + 145, c1_w, 850, fc="white", ec="#C8B7EF", lw=0.9, radius=10, z=1)
    label(ax, c1_x + 25, y + 180, "Fig. 4A/B", size=9.6, color="#332B73", weight="bold", ha="left")
    label(ax, c1_x + 25, y + 223, "distance-profile grouping", size=8.4, color="#332B73", ha="left")
    mini_profile(ax, c1_x + 45, y + 285, 640, 320, r"$d_B$ boundary distance", "db")
    mini_profile(ax, c1_x + 770, y + 285, 640, 320, r"$d_A$ lake/slump distance", "da")
    simple_box(ax, c1_x + 145, y + 690, 520, 78, "Transition / Interior", fc="#FFFFFF", ec=DU, color=DU, size=8.2, weight="bold")
    simple_box(ax, c1_x + 790, y + 690, 560, 78, "Lake/slump neighborhood\n/ Background", fc="#FFFFFF", ec=GRAD, color=GRAD, size=6.6, weight="bold")

    c2_x, c2_w = c1_x + c1_w + 45, 2050
    process_tables(ax, c2_x, y + 145, c2_w, 850)

    c3_x = c2_x + c2_w + 45
    c3_w = x + w - 65 - c3_x
    ale_panel_lower(ax, c3_x, y + 145, c3_w, 850)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = setup()
    ax.add_patch(patches.Rectangle((0, 0), W, H, fc="white", ec="none", zorder=-5))

    top_y, top_h = 55, 2520
    cols = [
        (45, top_y, 1000, top_h),
        (1080, top_y, 1220, top_h),
        (2335, top_y, 1240, top_h),
        (3610, top_y, 1580, top_h),
        (5225, top_y, 1930, top_h),
    ]
    data_sources(ax, *cols[0])
    insar_processing(ax, *cols[1])
    distance_registration(ax, *cols[2])
    conditioned_extremes(ax, *cols[3])
    susceptibility_panel(ax, *cols[4])

    for (x1, y1, w1, _h1), (x2, y2, _w2, _h2) in zip(cols[:-1], cols[1:]):
        arrow(ax, (x1 + w1 + 8, y1 + 1255), (x2 - 10, y2 + 1255), lw=1.8, ms=20, color="#616871")

    lower_band(ax)
    arrow(ax, (cols[2][0] + cols[2][2] / 2, cols[2][1] + cols[2][3] + 20), (1180, 2692), lw=0.95, ms=10, color="#616871")
    arrow(ax, (cols[3][0] + cols[3][2] / 2, cols[3][1] + cols[3][3] + 20), (3420, 2692), lw=0.95, ms=10, color="#616871")
    arrow(ax, (cols[4][0] + cols[4][2] / 2, cols[4][1] + cols[4][3] + 20), (5525, 2692), lw=0.95, ms=10, color="#616871")

    fig.savefig(PNG_OUT, dpi=600, metadata={"Software": "FigureS2_methodology_diagram.py"}, facecolor="white")
    with Image.open(PNG_OUT) as img:
        img.convert("RGB").save(PNG_OUT, dpi=(600, 600))
    with Image.open(PNG_OUT) as img:
        img.convert("RGB").save(PDF_OUT, "PDF", resolution=600.0)
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
