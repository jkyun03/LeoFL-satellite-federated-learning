# 지구·중력 상수와 좌표 변환(위경도→ECEF), 커버리지 디스크 생성 등 궤도 계산 보조 함수.
import numpy as np

# 상수 정의
G = 6.67430e-11  # 중력 상수 (m^3 kg^-1 s^-2)
M_earth = 5.972e24  # 지구 질량 (kg)
R_earth = 6371e3  # 지구 반지름 (m)


def shade(x, y, z):
    normal = np.stack([x, y, z], axis=2)
    normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)
    light_dir = np.array([1, 1, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.dot(normal, light_dir)
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    return intensity


def numpy_pairwise_combinations(x):
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]


def latlon_to_ecef(lat_deg, lon_deg):
    """
    위도/경도 → ECEF 좌표 (지구 중심 기준)
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R_earth * np.cos(lat) * np.cos(lon)
    y = R_earth * np.cos(lat) * np.sin(lon)
    z = R_earth * np.sin(lat)
    return np.array([x, y, z])


def orthonormal_basis(n):
    n = n / np.linalg.norm(n)
    if np.allclose(n, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(n, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def create_coverage_disk(center, normal, radius_km, resolution=100):
    """
    위성 커버리지 디스크 좌표를 생성하여 (X, Y, Z) 반환
    """
    center = np.asarray(center)
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)

    # normal과 수직인 두 벡터 u, v 생성
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # 극좌표 그리드
    theta = np.linspace(0, 2 * np.pi, resolution)
    r = np.linspace(0, radius_km, resolution)
    theta_grid, r_grid = np.meshgrid(theta, r)

    x_disk = r_grid * np.cos(theta_grid)
    y_disk = r_grid * np.sin(theta_grid)

    # 3D 좌표로 확장
    x_disk = x_disk[..., np.newaxis]  # (N, N, 1)
    y_disk = y_disk[..., np.newaxis]  # (N, N, 1)

    # 디스크 좌표 계산 (브로드캐스팅 안전)
    points = center + x_disk * u + y_disk * v  # (N, N, 3)
    X, Y, Z = points[..., 0], points[..., 1], points[..., 2]
    return X, Y, Z


def is_occluded(sat_pos, cam_dir):
    d = -cam_dir
    o = sat_pos
    to_center = -o
    proj_len = np.dot(to_center, d)

    if proj_len < 0:
        return False  # 위성이 지구 앞에 있음

    perp_dist = np.linalg.norm(np.cross(to_center, d)) / np.linalg.norm(d)
    return perp_dist < R_earth
