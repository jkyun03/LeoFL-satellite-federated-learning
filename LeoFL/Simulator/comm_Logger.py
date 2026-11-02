# 시뮬레이션 중 모델/오프로딩 데이터 전송량(bytes, hop 수)을 기록하는 통신량 로거.
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class CommLogger:
    seconds_per_round: int
    rows: list = field(default_factory=list)
    # columns: time, model_bytes, offload_bytes, model_hops, offload_hops, note

    def log(
        self,
        t: int,
        model_bytes: int = 0,
        offload_bytes: int = 0,
        model_hops: int = 0,
        offload_hops: int = 0,
        note: str = "",
    ):
        self.rows.append(
            (
                int(t),
                int(model_bytes),
                int(offload_bytes),
                int(model_hops),
                int(offload_hops),
                note,
            )
        )

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = pd.DataFrame(
            self.rows,
            columns=[
                "time",
                "model_bytes",
                "offload_bytes",
                "model_hops",
                "offload_hops",
                "note",
            ],
        )
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVE][COMM] {path} (rows={len(df)})")
        return df


def calc_model_bytes(model) -> int:
    total = 0
    for t in model.state_dict().values():
        total += t.numel() * t.element_size()
    return int(total)


def calc_sample_bytes(
    n_features: Optional[int] = None,
    sample_shape: Optional[Sequence[int]] = None,
    x_elem_size: int = 4,
    y_elem_size: int = 8,
) -> int:
    if n_features is not None:
        return int(n_features * x_elem_size + y_elem_size)

    if sample_shape is not None:
        num_elements = 1
        for dim in sample_shape:
            num_elements *= int(dim)
        return int(num_elements * x_elem_size + y_elem_size)

    raise ValueError("Either n_features or sample_shape must be provided.")
