# logs_eurosat/의 LH-FL 로그(리더·오프로딩·교환 기록)를 읽어 궤도면별 동작 개요도를 figures/에 그린다.
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# =========================================================
# User settings
# =========================================================
LOG_DIR = "./logs_eurosat"
PREFIX = "seed1_lhfl"

SAVE_DIR = "figures"
SAVE_NAME = f"{PREFIX}_lhfl_operation_overview"

MAX_PLANES_TO_SHOW = 3
SATS_TO_SHOW_PER_PLANE = 6
OFFLOADS_TO_SHOW_PER_PLANE = 2  # plane마다 표시할 offloading link 수
PREFER_SHORT_HOP_OFFLOAD = True  # hop이 짧은 offloading을 우선 표시

USE_HARDCODED_HOP_OFFLOAD = True


# =========================================================
# Style
# =========================================================
FIG_BG = "#FFFFFF"

PLANE_FACE = "#F1F5F9"
PLANE_EDGE = "#CBD5E1"

SAT_COLOR = "#1E3A8A"
SAT_EDGE = "white"

LEADER_COLOR = "#DC2626"
LEADER_EDGE = "#7F1D1D"

OFFLOAD_COLOR = "#F59E0B"
UPLOAD_COLOR = "#3B82F6"
BROADCAST_COLOR = "#14B8A6"
INTER_COLOR = "#14B8A6"

TEXT_COLOR = "#0F172A"
MUTED_TEXT = "#475569"

STEP1_FACE = "#FFF7ED"
STEP2_FACE = "#EFF6FF"
STEP3_FACE = "#EFF6FF"
STEP4_FACE = "#F0FDFA"

# =========================================================
# Load logs
# =========================================================
members_path = os.path.join(LOG_DIR, f"{PREFIX}_plane_members.csv")
leaders_path = os.path.join(LOG_DIR, f"{PREFIX}_leaders.csv")
exchange_path = os.path.join(LOG_DIR, f"{PREFIX}_leader_exchange.csv")
offload_path = os.path.join(LOG_DIR, f"{PREFIX}_offload_pairs.csv")

for path in [members_path, leaders_path, exchange_path, offload_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "먼저 LH-FL을 한 번 실행해서 overview 로그 CSV를 생성해야 합니다."
        )

members_df = pd.read_csv(members_path)
leaders_df = pd.read_csv(leaders_path)
exchange_df = pd.read_csv(exchange_path)
offload_df = pd.read_csv(offload_path)


# =========================================================
# Prepare representative planes
# =========================================================
plane_to_sats = {
    int(p): members_df[members_df["plane_id"] == p]
    .sort_values("local_idx")["sat_id"]
    .astype(int)
    .tolist()
    for p in sorted(members_df["plane_id"].unique())
}

leader_map = {
    int(row["plane_id"]): int(row["leader_sat_id"]) for _, row in leaders_df.iterrows()
}

# offloading이 많이 발생한 plane을 우선 표시
if not offload_df.empty and "moved_samples" in offload_df.columns:
    plane_scores = (
        offload_df.groupby("plane_id")["moved_samples"]
        .sum()
        .sort_values(ascending=False)
    )
    selected_planes = [int(p) for p in plane_scores.index[:MAX_PLANES_TO_SHOW]]
else:
    selected_planes = list(sorted(plane_to_sats.keys()))[:MAX_PLANES_TO_SHOW]

# 부족하면 앞 plane으로 채움
for p in sorted(plane_to_sats.keys()):
    if len(selected_planes) >= MAX_PLANES_TO_SHOW:
        break
    if p not in selected_planes:
        selected_planes.append(p)

selected_planes = selected_planes[:MAX_PLANES_TO_SHOW]


# =========================================================
# Utility functions
# =========================================================
def pick_display_sats(plane_id):
    """leader, offload src/dst를 우선 포함해 plane당 일부 satellite만 표시."""
    sats = list(plane_to_sats[plane_id])
    leader = leader_map[plane_id]

    important = [leader]

    rows = offload_df[offload_df["plane_id"] == plane_id].copy()
    if not rows.empty:
        if PREFER_SHORT_HOP_OFFLOAD and "hop" in rows.columns:
            rows = rows.sort_values(["hop", "moved_samples"], ascending=[True, False])
        else:
            rows = rows.sort_values("moved_samples", ascending=False)

        for _, r in rows.head(OFFLOADS_TO_SHOW_PER_PLANE).iterrows():
            important.append(int(r["src_sat_id"]))
            important.append(int(r["dst_sat_id"]))

    # 중복 제거
    display = []
    for s in important + sats:
        if s not in display:
            display.append(s)

    return display[:SATS_TO_SHOW_PER_PLANE]


def draw_curved_arrow(
    ax,
    xy1,
    xy2,
    color,
    lw=1.8,
    alpha=1.0,
    rad=0.15,
    linestyle="-",
    arrowstyle="->",
    mutation_scale=13,
    zorder=3,
    shrinkA=10,
    shrinkB=10,
):
    arrow = FancyArrowPatch(
        xy1,
        xy2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
    )
    ax.add_patch(arrow)


def draw_hop_path(
    ax,
    points,
    color,
    lw=2.2,
    alpha=0.95,
    rad=0.05,
    mutation_scale=13,
    zorder=5,
):
    """
    Draw multi-hop offloading path as consecutive short arrows.
    points: [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    if len(points) < 2:
        return

    for xy1, xy2 in zip(points[:-1], points[1:]):
        draw_curved_arrow(
            ax,
            xy1,
            xy2,
            color=color,
            lw=lw,
            alpha=alpha,
            rad=rad,
            arrowstyle="->",
            mutation_scale=mutation_scale,
            zorder=zorder,
            shrinkA=12,
            shrinkB=12,
        )


def offset_path(points, dx=0.0, dy=0.0):
    """Shift a path slightly to avoid visual overlap."""
    return [(x + dx, y + dy) for x, y in points]


# =========================================================
# Figure layout
# =========================================================
fig, ax = plt.subplots(figsize=(13.5, 7.2), facecolor=FIG_BG)
ax.set_facecolor(FIG_BG)

ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

plane_centers = [(2.4, 4.35), (7.0, 4.35), (11.6, 4.35)]
plane_w, plane_h = 3.15, 3.15

sat_positions = {}
leader_positions = {}

for idx, plane_id in enumerate(selected_planes):
    cx, cy = plane_centers[idx]

    # plane box
    box = FancyBboxPatch(
        (cx - plane_w / 2, cy - plane_h / 2),
        plane_w,
        plane_h,
        boxstyle="round,pad=0.18,rounding_size=0.25",
        linewidth=1.25,
        edgecolor=PLANE_EDGE,
        facecolor=PLANE_FACE,
        alpha=0.96,
        zorder=0,
    )
    ax.add_patch(box)

    plane_label = ["X", "Y", "Z"][idx]

    ax.text(
        cx,
        cy + plane_h / 2 + 0.38,
        f"Orbital plane {plane_label}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    # display satellites around an ellipse
    display_sats = pick_display_sats(plane_id)
    leader = leader_map[plane_id]

    # leader position fixed at top center
    leader_xy = (cx, cy + 1.05)
    sat_positions[leader] = leader_xy
    leader_positions[plane_id] = leader_xy

    # non-leader positions
    nonleaders = [s for s in display_sats if s != leader]
    # nonleader satellite positions
    # Important: pick_display_sats() returns offload src/dst pairs consecutively.
    # Therefore, positions are arranged in adjacent pairs so that offloading
    # arrows do not cross the whole plane box.
    base_positions = [
        (cx - 1.05, cy + 0.35),  # pair 1 source
        (cx - 0.95, cy - 0.72),  # pair 1 receiver
        (cx + 0.95, cy - 0.72),  # pair 2 source
        (cx + 1.05, cy + 0.35),  # pair 2 receiver
        (cx, cy - 1.05),  # extra satellite
    ]

    for s, pos in zip(nonleaders, base_positions):
        sat_positions[s] = pos

    # -------------------------------------------------
    # Intra-plane model upload as hop-to-hop paths
    # All displayed non-leader satellites are connected toward the leader.
    # Plane-level broadcast is omitted from the internal arrows
    # and is shown only as Step 4 to reduce clutter.
    # -------------------------------------------------
    left_top = (cx - 1.05, cy + 0.35)
    left_bottom = (cx - 0.95, cy - 0.72)
    bottom = (cx, cy - 1.05)
    right_bottom = (cx + 0.95, cy - 0.72)
    right_top = (cx + 1.05, cy + 0.35)

    # Hop-based upload paths toward leader.
    # These paths indicate that local model updates are relayed through
    # intra-plane communication paths rather than sent as direct one-hop links.
    upload_paths = [
        offset_path([left_bottom, left_top, leader_xy], dx=-0.03, dy=0.05),
        offset_path([bottom, left_bottom, left_top, leader_xy], dx=-0.01, dy=0.02),
        offset_path([right_bottom, right_top, leader_xy], dx=0.03, dy=0.05),
        offset_path([bottom, right_bottom, right_top, leader_xy], dx=0.01, dy=0.02),
    ]

    for path in upload_paths:
        draw_hop_path(
            ax,
            path,
            color=UPLOAD_COLOR,
            lw=1.20,
            alpha=0.42,
            rad=0.025,
            mutation_scale=9,
            zorder=1,
        )

    # representative offloading links in this plane
    # For visual clarity, show offloading as short intra-plane hop paths
    # rather than drawing direct source-to-destination chords.
    if USE_HARDCODED_HOP_OFFLOAD:
        # Schematic positions inside the current plane box
        left_top = (cx - 1.05, cy + 0.35)
        left_bottom = (cx - 0.95, cy - 0.72)
        bottom = (cx, cy - 1.05)
        right_bottom = (cx + 0.95, cy - 0.72)
        right_top = (cx + 1.05, cy + 0.35)

        # Offloading is drawn as lower/inner hop paths so that it does not
        # overlap with the blue satellite-to-leader upload paths.
        if idx == 0:
            offload_paths = [
                offset_path([left_top, left_bottom], dx=-0.12, dy=-0.04),
                offset_path([left_bottom, bottom, right_bottom], dy=-0.25),
                offset_path([right_top, right_bottom], dx=0.12, dy=-0.04),
            ]
        elif idx == 1:
            offload_paths = [
                offset_path([left_top, left_bottom], dx=-0.12, dy=-0.04),
                offset_path([right_top, right_bottom], dx=0.12, dy=-0.04),
            ]
        else:
            offload_paths = [
                offset_path([left_top, left_bottom], dx=-0.12, dy=-0.04),
                offset_path([left_bottom, bottom, right_bottom], dy=-0.25),
                offset_path([right_top, right_bottom], dx=0.12, dy=-0.04),
            ]

        for path in offload_paths:
            draw_hop_path(
                ax,
                path,
                color=OFFLOAD_COLOR,
                lw=2.05,
                alpha=0.86,
                rad=0.02,
                mutation_scale=12,
                zorder=5,
            )

        ax.text(
            cx,
            cy - 0.62,
            "\noffloading",
            ha="center",
            va="center",
            fontsize=8.8,
            color=OFFLOAD_COLOR,
            fontweight="bold",
            zorder=6,
        )

    else:
        rows = offload_df[offload_df["plane_id"] == plane_id].copy()

        if not rows.empty:
            if PREFER_SHORT_HOP_OFFLOAD and "hop" in rows.columns:
                rows = rows.sort_values(
                    ["hop", "moved_samples"], ascending=[True, False]
                )
            else:
                rows = rows.sort_values("moved_samples", ascending=False)

            drawn = 0
            rad_values = [0.22, -0.22, 0.18, -0.18]

            for _, r in rows.iterrows():
                if drawn >= OFFLOADS_TO_SHOW_PER_PLANE:
                    break

                src = int(r["src_sat_id"])
                dst = int(r["dst_sat_id"])

                if src == dst:
                    continue

                if src not in sat_positions or dst not in sat_positions:
                    continue

                rad = rad_values[drawn % len(rad_values)]

                draw_curved_arrow(
                    ax,
                    sat_positions[src],
                    sat_positions[dst],
                    color=OFFLOAD_COLOR,
                    lw=2.1,
                    alpha=0.9,
                    rad=rad,
                    arrowstyle="->",
                    mutation_scale=15,
                    zorder=4,
                    shrinkA=13,
                    shrinkB=13,
                )

                drawn += 1

            if drawn > 0:
                ax.text(
                    cx,
                    cy + 0.18,
                    "offloading",
                    ha="center",
                    fontsize=9.5,
                    color=OFFLOAD_COLOR,
                    fontweight="bold",
                    zorder=5,
                )

    # draw satellites
    for s in nonleaders:
        if s not in sat_positions:
            continue
        x, y = sat_positions[s]
        ax.scatter(
            x,
            y,
            s=160,
            marker="s",
            color=SAT_COLOR,
            edgecolor=SAT_EDGE,
            linewidth=1.2,
            zorder=6,
        )
        ax.text(
            x,
            y - 0.28,
            f"S{s + 1}",
            ha="center",
            va="top",
            fontsize=8.5,
            color=MUTED_TEXT,
            zorder=6,
        )

    # draw leader
    lx, ly = leader_xy
    ax.scatter(
        lx,
        ly,
        s=420,
        marker="*",
        color=LEADER_COLOR,
        edgecolor=LEADER_EDGE,
        linewidth=1.0,
        zorder=7,
    )
    ax.text(
        lx,
        ly + 0.48,
        f"Leader\nS{leader + 1}",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=TEXT_COLOR,
        fontweight="bold",
        linespacing=0.9,
        zorder=7,
    )

    ax.text(
        cx,
        cy - plane_h / 2 - 0.28,
        "intra-plane leader aggregation",
        ha="center",
        va="top",
        fontsize=9.2,
        color=UPLOAD_COLOR,
        fontweight="bold",
        zorder=8,
    )
# =========================================================
# Inter-plane leader exchange
# For overview clarity, connect the displayed leaders sequentially.
# This visualizes representative leader-to-leader decentralized mixing
# among the selected orbital planes.
# =========================================================
for a, b in zip(selected_planes[:-1], selected_planes[1:]):
    if a in leader_positions and b in leader_positions:
        draw_curved_arrow(
            ax,
            leader_positions[a],
            leader_positions[b],
            color=INTER_COLOR,
            lw=2.45,
            alpha=0.82,
            rad=0.18,
            linestyle=":",
            arrowstyle="<->",
            mutation_scale=16,
            zorder=2,
        )

ax.text(
    4.65,
    5.55,
    "inter-plane\nleader mixing",
    ha="center",
    va="center",
    fontsize=8.6,
    color=INTER_COLOR,
    fontweight="bold",
    linespacing=0.9,
    bbox=dict(
        boxstyle="round,pad=0.16",
        facecolor=FIG_BG,
        edgecolor="none",
        alpha=0.88,
    ),
    zorder=9,
)

# =========================================================
# Title and legend
# =========================================================
ax.text(
    7,
    7.45,
    "LH-FL operation overview",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=TEXT_COLOR,
)

ax.text(
    7,
    7.05,
    "Hop-based intra-plane upload, one-shot offloading, and inter-plane leader mixing",
    ha="center",
    va="center",
    fontsize=11.5,
    color=MUTED_TEXT,
)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="LEO satellite",
        markerfacecolor=SAT_COLOR,
        markeredgecolor=SAT_EDGE,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Leader satellite",
        markerfacecolor=LEADER_COLOR,
        markeredgecolor=LEADER_EDGE,
        markersize=14,
    ),
    Line2D([0], [0], color=UPLOAD_COLOR, lw=1.8, label="Satellite-to-leader upload"),
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.165),
    ncol=3,
    frameon=False,
    fontsize=9.6,
    handlelength=1.8,
    columnspacing=1.4,
    borderaxespad=0.0,
)

# =========================================================
# Save
# =========================================================
os.makedirs(SAVE_DIR, exist_ok=True)

png_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}.png")
pdf_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}.pdf")

plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

print(f"[SAVE] {png_path}")
print(f"[SAVE] {pdf_path}")

plt.show()
