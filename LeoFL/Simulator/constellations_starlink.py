# LeoFL/Simulator/constellations_starlink.py
# Starlink 유사 위성군 구성(궤도면·위성 수·고도·경사각) config 리스트를 생성한다.
from typing import List, Dict


def generate_starlink_like_constellation(
    num_planes: int = 12,
    sats_per_plane: int = 12,
    alt_km: float = 550.0,
    incl_deg: float = 53.0,
    orbital_period_min: float = 95.0,
) -> List[Dict]:
    """
    Starlink 1st shell 축소판.
    - 모든 위성: 같은 고도 / 같은 경사
    - plane마다 RAAN을 360/num_planes 간격으로 회전
    - 같은 plane 안 위성들은 offset_min 으로 true anomaly만 다르게
    """
    configs: List[Dict] = []

    raan_step = 360.0 / num_planes  # 궤도면 간 RAAN 간격

    for p in range(num_planes):
        raan_deg = p * raan_step

        for s in range(sats_per_plane):
            offset_min = (orbital_period_min / sats_per_plane) * s

            configs.append(
                {
                    "h_km": alt_km,
                    "incl_deg": incl_deg,
                    "offset_min": offset_min,
                    "plane_id": p,
                    "raan_deg": raan_deg,  # ★ 여기서 RAAN 부여
                }
            )

    return configs


STARLINK_LIKE_SMALL: List[Dict] = generate_starlink_like_constellation(
    num_planes=12,
    sats_per_plane=10,
    alt_km=550.0,
    incl_deg=53.0,
    orbital_period_min=95.0,
)
