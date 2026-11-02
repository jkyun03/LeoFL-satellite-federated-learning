# logs_eurosat/의 LH-FL 로그를 읽어 Walker 형태 3D 궤도 위에 리더·오프로딩·리더 간 교환을 시각화한다(figures/ 저장).
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

EARTH_COLOR = "#CFE8F3"
EARTH_ALPHA = 0.62

SAT_COLOR = "#243B8F"
SAT_EDGE = "white"
SAT_SIZE = 36
SAT_ALPHA = 0.96

LEADER_COLOR = "#C1121F"
LEADER_EDGE = "#111111"
LEADER_SIZE = 300

ORBIT_COLOR = "#A8A8A8"
ORBIT_LW = 0.85
ORBIT_ALPHA = 0.42

INTER_COLOR = "#1F78B4"
INTER_LW = 1.9
INTER_ALPHA = 0.72

OFFLOAD_COLOR = "#F28E2B"
OFFLOAD_LW = 1.6
OFFLOAD_ALPHA = 0.68

SHOW_OFFLOAD_LINKS = False
SHOW_OFFLOAD_LABELS = False
MAX_OFFLOAD_ARROWS = 0

# =========================================================
# User settings
# =========================================================
LOG_DIR = "./logs_eurosat"

# 보고 싶은 LH-FL 로그 prefix로 변경
# multi-seed 결과를 쓰면 seed1_lhfl 추천
# 기존 single run 결과를 쓰면 eurosat_offload 사용
PREFIX = "seed1_lhfl"

EARTH_RADIUS = 1.0
ORBIT_RADIUS = 2.4
PHASE_FACTOR = 1


# =========================================================
# Load CSV files
# =========================================================
members_path = os.path.join(LOG_DIR, f"{PREFIX}_plane_members.csv")
leaders_path = os.path.join(LOG_DIR, f"{PREFIX}_leaders.csv")
exchange_path = os.path.join(LOG_DIR, f"{PREFIX}_leader_exchange.csv")
offload_path = os.path.join(LOG_DIR, f"{PREFIX}_offload_pairs.csv")

required = [members_path, leaders_path, exchange_path, offload_path]
for path in required:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"scenarios.py 수정 후 LH-FL을 한 번 실행해야 생성됩니다."
        )

members_df = pd.read_csv(members_path)
leaders_df = pd.read_csv(leaders_path)
exchange_df = pd.read_csv(exchange_path)
offload_df = pd.read_csv(offload_path)

planes = sorted(members_df["plane_id"].unique().tolist())
num_planes = len(planes)

plane_to_sats = {
    int(p): members_df[members_df["plane_id"] == p]
    .sort_values("local_idx")["sat_id"]
    .astype(int)
    .tolist()
    for p in planes
}

leaders = leaders_df["leader_sat_id"].astype(int).tolist()

leader_edges = []
for _, row in exchange_df.iterrows():
    leader_edges.append((int(row["src_leader"]), int(row["dst_leader"])))

offload_pairs = []
for _, row in offload_df.iterrows():
    offload_pairs.append(
        (
            int(row["src_sat_id"]),
            int(row["dst_sat_id"]),
            int(row["moved_samples"]),
        )
    )

# moved_samples가 큰 상위 offloading만 표시
if MAX_OFFLOAD_ARROWS is not None and len(offload_pairs) > MAX_OFFLOAD_ARROWS:
    offload_pairs = sorted(offload_pairs, key=lambda x: x[2], reverse=True)
    offload_pairs = offload_pairs[:MAX_OFFLOAD_ARROWS]
# =========================================================
# Generate Walker-like 3D positions from experimental plane structure
# =========================================================


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def orbit_point(radius, true_anomaly, raan, inclination):
    p_orb = np.array(
        [
            radius * np.cos(true_anomaly),
            radius * np.sin(true_anomaly),
            0.0,
        ]
    )

    return rot_z(raan) @ rot_x(inclination) @ p_orb


def build_positions(plane_to_sats, orbit_radius, phase_factor=1, inclination_deg=53.0):
    sat_pos = {}
    orbit_curves = {}

    inclination = np.deg2rad(inclination_deg)
    t_curve = np.linspace(0, 2 * np.pi, 400)

    sorted_planes = sorted(plane_to_sats.keys())
    n_planes = len(sorted_planes)

    for plane_order, p in enumerate(sorted_planes):
        sat_ids = plane_to_sats[p]
        sats_per_plane = len(sat_ids)

        raan = 2 * np.pi * plane_order / max(n_planes, 1)

        curve = np.array(
            [orbit_point(orbit_radius, t, raan, inclination) for t in t_curve]
        )
        orbit_curves[p] = curve

        for local_idx, sat_id in enumerate(sat_ids):
            phase_offset = (
                2
                * np.pi
                * phase_factor
                * plane_order
                / max(n_planes * sats_per_plane, 1)
            )

            true_anomaly = 2 * np.pi * local_idx / max(sats_per_plane, 1) + phase_offset

            sat_pos[int(sat_id)] = orbit_point(
                orbit_radius,
                true_anomaly,
                raan,
                inclination,
            )

    return sat_pos, orbit_curves


sat_pos, orbit_curves = build_positions(
    plane_to_sats=plane_to_sats,
    orbit_radius=ORBIT_RADIUS,
    phase_factor=PHASE_FACTOR,
)


# =========================================================
# Plot
# =========================================================
fig = plt.figure(figsize=(10.8, 9.2), facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# Earth
u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)
x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(
    x,
    y,
    z,
    color=EARTH_COLOR,
    alpha=EARTH_ALPHA,
    linewidth=0,
    shade=True,
)

# Orbit curves
for p, curve in orbit_curves.items():
    ax.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        linestyle="--",
        linewidth=ORBIT_LW,
        color=ORBIT_COLOR,
        alpha=ORBIT_ALPHA,
    )

# Satellites
all_sat_ids = sorted(sat_pos.keys())
all_xyz = np.array([sat_pos[s] for s in all_sat_ids])

ax.scatter(
    all_xyz[:, 0],
    all_xyz[:, 1],
    all_xyz[:, 2],
    marker="s",
    s=SAT_SIZE,
    color=SAT_COLOR,
    edgecolor=SAT_EDGE,
    linewidth=0.55,
    alpha=SAT_ALPHA,
    depthshade=True,
)

# Leaders
leader_xyz = np.array([sat_pos[s] for s in leaders if s in sat_pos])

ax.scatter(
    leader_xyz[:, 0],
    leader_xyz[:, 1],
    leader_xyz[:, 2],
    marker="*",
    s=LEADER_SIZE,
    color=LEADER_COLOR,
    edgecolor=LEADER_EDGE,
    linewidth=0.9,
    depthshade=False,
)

# Leader exchange edges
for src, dst in leader_edges:
    if src not in sat_pos or dst not in sat_pos:
        continue

    a = sat_pos[src]
    b = sat_pos[dst]

    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        [a[2], b[2]],
        linestyle=":",
        linewidth=INTER_LW,
        color=INTER_COLOR,
        alpha=INTER_ALPHA,
    )

# Offloading arrows
for src, dst, moved in offload_pairs:
    if src not in sat_pos or dst not in sat_pos:
        continue

    a = sat_pos[src]
    b = sat_pos[dst]
    vec = b - a

    ax.quiver(
        a[0],
        a[1],
        a[2],
        vec[0],
        vec[1],
        vec[2],
        color=OFFLOAD_COLOR,
        linewidth=OFFLOAD_LW,
        arrow_length_ratio=0.09,
        alpha=OFFLOAD_ALPHA,
    )

    # Offloading arrows
    if SHOW_OFFLOAD_LINKS:
        for src, dst, moved in offload_pairs:
            if src not in sat_pos or dst not in sat_pos:
                continue

            a = sat_pos[src]
            b = sat_pos[dst]
            vec = b - a

            ax.quiver(
                a[0],
                a[1],
                a[2],
                vec[0],
                vec[1],
                vec[2],
                color=OFFLOAD_COLOR,
                linewidth=OFFLOAD_LW,
                arrow_length_ratio=0.09,
                alpha=OFFLOAD_ALPHA,
            )


# Legend
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="LEO satellite",
        markerfacecolor=SAT_COLOR,
        markeredgecolor=SAT_EDGE,
        markersize=9,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Leader satellite",
        markerfacecolor=LEADER_COLOR,
        markeredgecolor=LEADER_EDGE,
        markersize=13,
    ),
    Line2D(
        [0],
        [0],
        color=ORBIT_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Intra-plane orbit / ISL",
    ),
    Line2D(
        [0],
        [0],
        color=INTER_COLOR,
        linestyle=":",
        linewidth=2.6,
        label="Inter-plane leader exchange",
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.07),
    ncol=2,
    frameon=False,
    fontsize=12,
    handlelength=2.2,
    columnspacing=1.5,
)

# Aesthetics
ax.view_init(elev=24, azim=-48)

lim = ORBIT_RADIUS * 1.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim * 0.85, lim * 0.85)

# 3D grid / axes for constellation layout
ax.set_box_aspect([1, 1, 0.85])

# Keep light 3D grid, but hide axis labels/tick labels
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.tick_params(axis="both", which="major", length=0, pad=0)
ax.grid(True)

# light background panes
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.pane.set_edgecolor("#D0D0D0")

# light grid lines
ax.xaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.65)
ax.yaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.65)
ax.zaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.65)

ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
ax.zaxis._axinfo["grid"]["linewidth"] = 0.6

plt.tight_layout()

os.makedirs("figures", exist_ok=True)
png_path = os.path.join("figures", f"{PREFIX}_lhfl_overview_3d.png")
pdf_path = os.path.join("figures", f"{PREFIX}_lhfl_overview_3d.pdf")

plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

print(f"[SAVE] {png_path}")
print(f"[SAVE] {pdf_path}")

plt.show()
