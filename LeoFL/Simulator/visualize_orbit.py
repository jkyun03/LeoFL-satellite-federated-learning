# 고정/Starlink-like 위성군의 궤도와 특정 시점 위성 위치를 Plotly 3D로 시각화한다.
import numpy as np
import plotly.graph_objects as go
from LeoFL.Simulator.fl_core import (
    build_fixed_constellation,
    FIXED_CONSTELLATION,
    compute_orbits_for_rounds,
)

from LeoFL.Simulator.constellations_starlink import STARLINK_LIKE_SMALL

# ---- 설정 ----
VISUALIZE_TIME_SEC = 0  # 보기 원하는 시간(sec)
SECONDS_PER_ROUND = 60  # 궤도 스텝 크기(sec)

EARTH_RADIUS_KM = 6371.0
ISL_DIST_THRESHOLD = 2000.0


def _sec_to_frame(t_sec: int, seconds_per_round: int, num_rounds: int) -> int:
    frame = t_sec // seconds_per_round
    if frame >= num_rounds:
        frame = num_rounds - 1
    return int(frame)


def plot_satellite_constellation():

    print("위성 궤도 계산 중...")

    # ==== Starlink-like 옵션 ====
    USE_STARLINK = False

    if USE_STARLINK:
        sat_configs = STARLINK_LIKE_SMALL
        print(
            f"Starlink-like constellation 사용 (총 {len(STARLINK_LIKE_SMALL)}개 위성)"
        )
    else:
        sat_configs = FIXED_CONSTELLATION
        print(f"기존 FIXED constellation 사용 (총 {len(FIXED_CONSTELLATION)} 개 위성)")
    # ===========================

    sats = build_fixed_constellation(sat_configs)

    # ▶ 시각화하려는 시간이 보장되도록 스텝 확보
    min_rounds_for_vis = VISUALIZE_TIME_SEC // SECONDS_PER_ROUND + 1
    num_rounds = max(
        200, min_rounds_for_vis
    )  # Starlink-like면 궤도 궤적이 많으므로 200 이상 추천

    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=SECONDS_PER_ROUND
    )

    # -------------------------------
    # Plotly Figure 생성
    # -------------------------------
    fig = go.Figure()

    # --- 지구 ---
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    x = EARTH_RADIUS_KM * np.outer(np.cos(theta), np.sin(phi))
    y = EARTH_RADIUS_KM * np.outer(np.sin(theta), np.sin(phi))
    z = EARTH_RADIUS_KM * np.outer(np.ones(50), np.cos(phi))

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="Blues",
            showscale=False,
            opacity=0.3,
            name="Earth",
        )
    )

    # --- 궤도 궤적 ---
    plane_ids = [cfg["plane_id"] for cfg in sat_configs]
    unique_planes = sorted(set(plane_ids))

    # 8개 plane 대응 색상 팔레트 확장
    color_list = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "cyan",
        "gold",
        "lime",
    ]
    plane_colors = {p: color_list[p % len(color_list)] for p in unique_planes}

    plane_seen = set()

    for idx, sat in enumerate(sats):
        pid = plane_ids[idx]
        color = plane_colors[pid]

        traj_x, traj_y, traj_z = [], [], []
        for frame in range(num_rounds):
            pos = sat.get_leo_orbit_position(frame)
            pos_km = np.array(pos) / 1000.0
            traj_x.append(pos_km[0])
            traj_y.append(pos_km[1])
            traj_z.append(pos_km[2])

        show_legend = pid not in plane_seen
        if show_legend:
            plane_seen.add(pid)

        fig.add_trace(
            go.Scatter3d(
                x=traj_x,
                y=traj_y,
                z=traj_z,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=show_legend,
                name=f"Plane {pid} orbit",
                hoverinfo="skip",
            )
        )

    # --- 현재 시점 위성 위치 ---
    frame_idx = _sec_to_frame(VISUALIZE_TIME_SEC, SECONDS_PER_ROUND, num_rounds)

    sat_positions = []
    for sat in sats:
        pos = sat.get_leo_orbit_position(frame_idx)
        sat_positions.append(np.array(pos) / 1000.0)
    sat_positions = np.array(sat_positions)

    sat_x = sat_positions[:, 0]
    sat_y = sat_positions[:, 1]
    sat_z = sat_positions[:, 2]

    marker_colors = [plane_colors[p] for p in plane_ids]

    fig.add_trace(
        go.Scatter3d(
            x=sat_x,
            y=sat_y,
            z=sat_z,
            mode="markers",
            marker=dict(size=5, color=marker_colors, symbol="diamond"),
            text=[f"Sat {i} (Plane {plane_ids[i]})" for i in range(len(sat_positions))],
            name="Satellites",
        )
    )

    # --- 레이아웃 ---
    fig.update_layout(
        title="Starlink-Like Satellite Constellation Visualization",
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    print("시각화 생성 완료!")
    fig.show()


if __name__ == "__main__":
    plot_satellite_constellation()
