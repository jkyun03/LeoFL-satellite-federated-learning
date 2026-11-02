# Walker-Delta 궤도 구성과 LH-FL 리더/오프로딩/교환 관계를 개념적으로 보여주는 3D 개요도를 그린다.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

# =========================================================
# [A] 실험 환경에 맞게 수정할 부분
# =========================================================
NUM_PLANES = 8
SATS_PER_PLANE = 10
EARTH_RADIUS = 1.0
ORBIT_RADIUS = 2.2

# 예시: sat 0~9 -> plane 0, 10~19 -> plane 1 ...
plane_of_sat = {}
for p in range(NUM_PLANES):
    for i in range(SATS_PER_PLANE):
        sid = p * SATS_PER_PLANE + i
        plane_of_sat[sid] = p

# 실제 leader 결과 넣기
leaders = [3, 12, 26, 35, 41, 54, 68, 73]

# 실제 offloading 결과 넣기: (sender, receiver, moved_samples)
offload_pairs = [
    (1, 3, 25),
    (8, 3, 18),
    (11, 12, 22),
    (24, 26, 30),
    (31, 35, 15),
]

# 실제 leader 간 교환 결과 넣기
leader_edges = [
    (3, 12),
    (12, 26),
    (26, 35),
    (35, 41),
    (41, 54),
    (54, 68),
    (68, 73),
    (73, 3),
]


# =========================================================
# [B] 위성 위치 생성 (방법 B: 실험 기반 개념 배치)
# =========================================================
def generate_satellite_positions(num_planes, sats_per_plane, orbit_radius):
    sat_pos = {}
    plane_angles = np.linspace(0, np.pi, num_planes, endpoint=False)

    for p in range(num_planes):
        phi = plane_angles[p]
        thetas = np.linspace(0, 2 * np.pi, sats_per_plane, endpoint=False)

        for i, theta in enumerate(thetas):
            sid = p * sats_per_plane + i

            # 기본 원형 궤도 (x-z 평면)
            x0 = orbit_radius * np.cos(theta)
            y0 = 0
            z0 = orbit_radius * np.sin(theta)

            # y축 기준 회전 -> plane마다 방향 다르게
            x = x0 * np.cos(phi) + y0 * np.sin(phi)
            y = -x0 * np.sin(phi) + y0 * np.cos(phi)
            z = z0

            sat_pos[sid] = np.array([x, y, z])

    return sat_pos


sat_pos = generate_satellite_positions(NUM_PLANES, SATS_PER_PLANE, ORBIT_RADIUS)

# =========================================================
# [C] 3D Figure
# =========================================================
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection="3d")

# ----- Earth -----
u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)
x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(x, y, z, color="skyblue", alpha=0.55, linewidth=0)

# ----- Plane orbit lines -----
for p in range(NUM_PLANES):
    pts = []
    for i in range(SATS_PER_PLANE):
        sid = p * SATS_PER_PLANE + i
        pts.append(sat_pos[sid])
    pts.append(pts[0])
    pts = np.array(pts)

    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        linestyle="--",
        linewidth=1.3,
        color="gray",
        alpha=0.8,
    )

# ----- All satellites -----
all_xyz = np.array([sat_pos[s] for s in sorted(sat_pos.keys())])
ax.scatter(
    all_xyz[:, 0],
    all_xyz[:, 1],
    all_xyz[:, 2],
    s=35,
    c="royalblue",
    marker="s",
    edgecolors="white",
)

# ----- Leaders -----
leader_xyz = np.array([sat_pos[s] for s in leaders])
ax.scatter(
    leader_xyz[:, 0],
    leader_xyz[:, 1],
    leader_xyz[:, 2],
    s=280,
    c="red",
    marker="*",
    edgecolors="black",
    linewidths=0.9,
)

# ----- Leader-to-leader exchange -----
for a, b in leader_edges:
    pa, pb = sat_pos[a], sat_pos[b]
    ax.plot(
        [pa[0], pb[0]],
        [pa[1], pb[1]],
        [pa[2], pb[2]],
        linestyle=":",
        linewidth=2.4,
        color="dodgerblue",
        alpha=0.95,
    )

# ----- One-shot offloading -----
for src, dst, moved in offload_pairs:
    ps, pd = sat_pos[src], sat_pos[dst]
    vec = pd - ps

    ax.quiver(
        ps[0],
        ps[1],
        ps[2],
        vec[0],
        vec[1],
        vec[2],
        arrow_length_ratio=0.08,
        color="darkorange",
        linewidth=2.2,
    )

    mid = (ps + pd) / 2
    ax.text(
        mid[0],
        mid[1],
        mid[2],
        f"+{moved}",
        fontsize=12,
        color="darkorange",
        weight="bold",
    )

# =========================================================
# [D] 보기 좋게 설정
# =========================================================
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()
ax.view_init(elev=20, azim=35)

# 여백
plt.subplots_adjust(bottom=0.12, top=0.93)

# 범례 크게
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="LEO satellite",
        markerfacecolor="royalblue",
        markeredgecolor="white",
        markersize=12,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Leader satellite",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=18,
    ),
    Line2D(
        [0], [0], color="gray", lw=2, linestyle="--", label="Intra-plane orbit / ISL"
    ),
    Line2D(
        [0],
        [0],
        color="dodgerblue",
        lw=3,
        linestyle=":",
        label="Inter-plane leader exchange",
    ),
    Line2D([0], [0], color="darkorange", lw=3, label="One-shot offloading"),
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=2,
    frameon=False,
    fontsize=15,
)

# 저장
plt.savefig("walker_overview_3d.png", dpi=300, bbox_inches="tight")
plt.show()
