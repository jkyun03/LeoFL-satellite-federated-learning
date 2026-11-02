# LEO 위성 1기의 궤도(고도·경사각·RAAN) 위치 계산과 지상 지점 커버리지 판정을 담당하는 클래스.
import numpy as np
from .misc import *


class LEOSat:
    def __init__(self, h_km, offset_sec, incl_deg, id, raan_deg=0.0):
        self.id = id
        self.h_km = h_km
        self.offset_sec = offset_sec
        self.incl_deg = incl_deg
        self.orbit_path = None
        self.satellite = None
        self.raan_deg = raan_deg

        self.x = None
        self.y = None
        self.y = None
        self.z = None

    def compute_leo_orbit(self, times=[]):
        # 단위 변환
        h = self.h_km * 1000
        r = R_earth + h

        # 궤도 속도 및 주기 계산
        v = np.sqrt(G * M_earth / r)
        T = 2 * np.pi * r / v  # 주기 (초)
        # print(f"위성 고도: {self.h_km} km")
        # print(f"공전 속도: {v:.2f} m/s")
        # print(f"공전 주기: {T/60:.2f} 분")

        self.theta = 2 * np.pi * (times + self.offset_sec) / T

        # 라디안 변환
        incl_rad = np.radians(self.incl_deg)
        raan_rad = np.radians(self.raan_deg)

        # ★ RAAN + inclination 반영한 3D 좌표
        # argument of perigee ω = 0 가정
        cosO = np.cos(raan_rad)
        sinO = np.sin(raan_rad)
        cosI = np.cos(incl_rad)
        sinI = np.sin(incl_rad)
        cosT = np.cos(self.theta)
        sinT = np.sin(self.theta)

        self.x = r * (cosO * cosT - sinO * sinT * cosI)
        self.y = r * (sinO * cosT + cosO * sinT * cosI)
        self.z = r * (sinT * sinI)

        """
        self.x = r * np.cos(self.theta)
        self.y = r * np.sin(self.theta)

        incl_rad = np.radians(self.incl_deg)
        self.z = self.y * np.sin(incl_rad)
        self.y = self.y * np.cos(incl_rad)
        """

    def get_leo_orbit_position(self, frame):
        return self.x[frame], self.y[frame], self.z[frame]

    def prepare_visualization(self, ax):
        # self.orbit_path, = ax.plot(self.x / 1000, self.y / 1000, linestyle='--', color='gray', alpha=0.3)
        (self.satellite,) = ax.plot([], [], [], "ro", label="위성%s" % self.id)

    def init_satellite_visualiztion(self):
        self.satellite.set_data([], [])
        self.satellite.set_3d_properties([])

    def proceed_satellite_visualiztion(self, frame, cam_dir=None):
        self.satellite.set_data([self.x[frame] / 1000], [self.y[frame] / 1000])
        self.satellite.set_3d_properties([self.z[frame] / 1000])
        if cam_dir is not None:
            occluded = is_occluded(
                np.array(self.get_leo_orbit_position(frame)), cam_dir
            )
            color = "gray" if occluded else "red"
            alpha = 0.3 if occluded else 1.0
            self.satellite.set_markerfacecolor(color)
            self.satellite.set_alpha(alpha)


def is_point_in_coverage(sat_pos, lat_deg, lon_deg):
    """
    지표상의 위도/경도 좌표가 위성 커버리지 안에 있는지 판단

    Parameters:
        sat_pos_km (array-like): 위성 위치 (x, y, z) in km
        lat_deg, lon_deg (float): 관측 지점 위도/경도 (degrees)
        R_km (float): 지구 반지름 (기본 6371 km)

    Returns:
        (bool, float, float): (커버리지 여부, 중심각 θ, 최대 중심각 α) in degrees
    """
    sat_pos = np.array(sat_pos).reshape(-1)
    r_sat = np.linalg.norm(sat_pos)

    # 커버리지 최대 중심각 (coverage half-angle)
    alpha_rad = np.arccos(R_earth / r_sat)

    # 지표 지점 위치 벡터
    point_vec = latlon_to_ecef(lat_deg, lon_deg)

    # 중심각 계산
    cos_theta = np.dot(sat_pos, point_vec) / (r_sat * R_earth)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)

    inside = theta_rad <= alpha_rad
    return inside, np.degrees(theta_rad), np.degrees(alpha_rad)


def calculate_angle_between_satellites(pos1, pos2, in_degrees=True):
    """
    두 위성 위치 벡터 간의 중심각(지구 중심 기준)을 계산합니다.

    Parameters:
        pos1 (array-like): 위성 A의 3차원 위치 (x, y, z)
        pos2 (array-like): 위성 B의 3차원 위치 (x, y, z)
        in_degrees (bool): True이면 도(degree) 단위로 반환, False이면 라디안

    Returns:
        각도 (float): 두 벡터 간의 각도
    """
    # 벡터를 numpy 배열로 변환
    a = np.array(pos1)
    b = np.array(pos2)

    # 벡터 크기
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 벡터 내적
    dot_product = np.dot(a, b)

    # arccos 계산 (클램핑 필수!)
    cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)

    if in_degrees:
        return np.degrees(theta_rad)
    else:
        return theta_rad


def check_los(pos1, pos2):
    if enumerate(pos1) == enumerate(pos2):
        return 1
    theta = calculate_angle_between_satellites(pos1[:-1], pos2[:-1])
    theta_crit = np.degrees(
        np.arccos(R_earth / (R_earth + pos1[-1] * 1000))
        + np.arccos(R_earth / (R_earth + pos2[-1] * 1000))
    )
    return theta <= theta_crit
