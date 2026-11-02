# 여러 LEO 위성의 궤도를 3D 애니메이션으로 시뮬레이션하는 초기 시각화 스크립트.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import pairwise_distances

from LeoFL.Leo.leosat import *
from LeoFL.Leo.misc import *
from LeoFL.Leo.community import update_community_visualiztion
import sys

np.set_printoptions(threshold=sys.maxsize)

# from matplotlib import rc
# rc('font', family='AppleGothic')
# plt.rcParams['axes.unicode_minus'] = False

GS_lat = 37.5665
GS_lon = 126.9780

global conn_time, conn_cnt, conn_ts


def simulate_leo_orbit(sat_list, duration_min=10, timestep_sec=10):
    """
    LEO 위성 궤도 시뮬레이션 및 시각화

    Parameters:
        h_km (float): 지표면으로부터의 위성 고도 (km)
        duration_min (int): 애니메이션 총 시간 (분)
        timestep_sec (int): 프레임당 시간 간격 (초)
    """
    # 시간 배열 생성
    total_time_sec = duration_min * 60
    times = np.arange(0, total_time_sec, timestep_sec)

    for sat in sat_list:
        sat.compute_leo_orbit(times)
    h_vector = np.array([sat.h_km for sat in sat_list])
    max_h_km = np.max(h_vector)

    # 애니메이션 준비
    fig = plt.figure(figsize=(9, 9))
    fig_comm, ax_comm = plt.subplots(2, figsize=(6, 8))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-(R_earth / 1000 + max_h_km) * 1.2, (R_earth / 1000 + max_h_km) * 1.2)
    ax.set_ylim(-(R_earth / 1000 + max_h_km) * 1.2, (R_earth / 1000 + max_h_km) * 1.2)
    ax.set_zlim(-(R_earth / 1000 + max_h_km) * 1.2, (R_earth / 1000 + max_h_km) * 1.2)
    ax.grid(True)
    # ax.set_title('LEO 위성 궤도 애니메이션')
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")

    # 지구 구체 좌표
    phi, theta_s = np.mgrid[0 : np.pi : 50j, 0 : 2 * np.pi : 50j]
    X_sphere = R_earth * np.sin(phi) * np.cos(theta_s)
    Y_sphere = R_earth * np.sin(phi) * np.sin(theta_s)
    Z_sphere = R_earth * np.cos(phi)

    # GS
    ground_station = latlon_to_ecef(GS_lat, GS_lon) / 1000.0
    gs_plot = ax.scatter(*ground_station, color="green", s=80, label="기지국")

    # 지구
    shade_colors = shade(X_sphere, Y_sphere, Z_sphere)
    ax.plot_surface(
        X_sphere / 1000,
        Y_sphere / 1000,
        Z_sphere / 1000,
        facecolors=plt.cm.Blues(shade_colors),
        rstride=4,
        cstride=4,
        linewidth=0,
        antialiased=False,
        alpha=0.3,
    )

    (sample_coverage_patch,) = ax.plot([], [], [], "g-", lw=2, label="커버리지")

    for sat in sat_list:
        sat.prepare_visualization(ax)

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # 초기화 함수
    def init():
        global conn_time, conn_cnt, conn_ts, prev_GS_connectivity, prev_ts
        for sat in sat_list:
            sat.init_satellite_visualiztion()
        sample_coverage_patch.set_data([], [])
        sample_coverage_patch.set_3d_properties([])
        time_text.set_text("")
        conn_time = np.zeros((len(sat_list) + 1, len(sat_list) + 1))
        conn_cnt = np.zeros((len(sat_list) + 1, len(sat_list) + 1))
        conn_ts = np.zeros((len(sat_list) + 1, len(sat_list) + 1))
        prev_GS_connectivity = np.array([0] * len(sat_list))
        prev_GS_connectivity = np.array([0] * len(sat_list))
        prev_ts = 0

    # 프레임 업데이트 함수
    def update(frame):
        global conn_time, conn_ts, prev_GS_connectivity, prev_ts
        # 현재 카메라 시점으로부터 관측 방향 추출
        # azim = np.radians(ax.azim)
        # elev = np.radians(ax.elev)
        cam_dir = None
        # np.array([
        #    np.cos(elev) * np.cos(azim),
        #    np.cos(elev) * np.sin(azim),
        #    np.sin(elev)
        # ])

        for sat in sat_list:
            sat.proceed_satellite_visualiztion(frame, cam_dir=cam_dir)

        # sample coverage 계산
        sat_xyz = [sat_list[0].get_leo_orbit_position(frame)]
        r_sat = np.linalg.norm(sat_xyz)
        alpha = np.arccos(R_earth / r_sat)
        r_cov = R_earth * np.sin(alpha)
        center_cov = sat_xyz / r_sat * (R_earth * np.cos(alpha))
        # X_cov, Y_cov, Z_cov = create_coverage_disk(center_cov / 1000, sat_xyz / r_sat, r_cov / 1000)
        # 업데이트 시 patch 대체
        # if coverage_surf is not None:
        #    coverage_surf.remove()  # 이전 프레임 제거
        # coverage_surf = ax.plot_surface(X_cov, Y_cov, Z_cov,
        #                                color='green', alpha=0.3, shade=True, linewidth=0)

        u_dir, v_dir = orthonormal_basis(sat_xyz)
        theta_cov = np.linspace(0, 2 * np.pi, 100)
        circle_pts = np.outer(np.cos(theta_cov), u_dir) + np.outer(
            np.sin(theta_cov), v_dir
        )
        circle_pts *= r_cov
        circle_pts += center_cov
        sample_coverage_patch.set_data(circle_pts[:, 0] / 1000, circle_pts[:, 1] / 1000)
        sample_coverage_patch.set_3d_properties(circle_pts[:, 2] / 1000)

        GS_connectivity = np.array(
            [
                int(
                    is_point_in_coverage(
                        sat.get_leo_orbit_position(frame), GS_lat, GS_lon
                    )[0]
                )
                for sat in sat_list
            ]
        )
        if np.sum(prev_GS_connectivity != GS_connectivity) > 0:
            print(prev_GS_connectivity, frame - prev_ts)
            prev_ts = frame
            prev_GS_connectivity = GS_connectivity
        # print(GS_connectivity)

        pos_matrix = np.array([sat.get_leo_orbit_position(frame) for sat in sat_list])
        temp = np.concatenate((pos_matrix, h_vector.reshape(-1, 1)), axis=1)
        adj_matrix = pairwise_distances(temp, metric=check_los)
        dist_matrix = pairwise_distances(temp)
        print(np.max(dist_matrix))
        dist_matrix = dist_matrix / np.max(dist_matrix)
        print(adj_matrix * dist_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix_with_gs = np.concatenate(
            (GS_connectivity.reshape(1, -1), adj_matrix), axis=0
        )
        adj_matrix_with_gs = np.concatenate(
            (np.append([0], GS_connectivity).reshape(-1, 1), adj_matrix_with_gs), axis=1
        )

        lower = np.tril(conn_time)
        upper = np.triu(conn_time)
        conn_cnt[(upper > 0) * (adj_matrix_with_gs == 0)] += 1
        conn_ts[np.transpose((upper > 0) * (adj_matrix_with_gs == 0))] += frame

        conn_time[adj_matrix_with_gs == 0] = 0
        conn_time[lower > 0] = lower[lower > 0]
        # lower = np.tril(conn_time)
        condition1 = (
            (lower == 1)
            * (np.transpose(upper) != lower)
            * np.transpose((upper > 0) * (adj_matrix_with_gs == 0))
        )
        condition2 = (
            (lower > 1)
            * (np.transpose(upper) != lower)
            * np.transpose((upper > 0) * (adj_matrix_with_gs == 0))
        )
        # print((lower > 0))
        # print((np.transpose(upper) != lower))
        # print(condition)
        conn_time[condition1] = np.transpose(upper)[condition1]
        conn_time[condition2] = (
            0.8 * lower[condition2] + 0.2 * np.transpose(upper)[condition2]
        )
        # print(np.transpose(upper)[condition])
        conn_time[lower == 0] = (
            conn_time[lower == 0] + adj_matrix_with_gs[lower == 0]
        )  # np.triu(adj_matrix_with_gs) + 0.2 * np.tril(adj_matrix_with_gs)
        # print(np.transpose(conn_cnt))
        # print(conn_time)
        upper = np.triu(conn_time)
        conn_ts[upper == 1] = frame
        # print((np.tril(conn_time) / np.tril(np.transpose(conn_cnt))))

        # print(adj_matrix_with_gs)
        update_community_visualiztion(adj_matrix_with_gs, fig_comm, ax_comm)
        minutes = times[frame] // 60
        seconds = times[frame] % 60
        time_text.set_text(f"{int(minutes)}m {int(seconds)}s")

    ani = FuncAnimation(
        fig, update, frames=len(times), init_func=init, interval=50, repeat=False
    )

    # plt.legend()
    plt.show()
    # writer = animation.PillowWriter(fps=30)
    # ani.save("test.gif", writer=writer)


if __name__ == "__main__":
    sat_list = []
    np.random.seed(1)
    for i in range(0, 50):
        h_km = np.random.randint(500, 2500)
        offset_sec = np.random.randint(0, 60 * 120)
        incl_deg = np.random.randint(0, 90)
        sat_list.append(
            LEOSat(h_km=h_km, offset_sec=offset_sec, incl_deg=incl_deg, id=i + 1)
        )
    # 실행
    simulate_leo_orbit(sat_list, duration_min=1200, timestep_sec=60)
