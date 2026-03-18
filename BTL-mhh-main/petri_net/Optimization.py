import collections
from typing import Tuple, List, Optional
from pyeda.inter import *
from collections import deque
import numpy as np

def max_reachable_marking(
    place_ids: List[str], 
    bdd: BinaryDecisionDiagram, 
    c: np.ndarray
) -> Tuple[Optional[List[int]], Optional[int]]:
  
    # Chuẩn hóa vector trọng số
    c = np.asarray(c).flatten()
    num_places = len(place_ids)
    if c.shape[0] != num_places:
        raise ValueError(
            f"Độ dài vector c = {c.shape[0]} không khớp số place = {num_places}"
        )

    # Nếu BDD rỗng thì không có marking reachable
    if bdd.is_zero():
        return None, None

    # Tạo BDD variables tương ứng với từng place
    place_vars = [bddvar(pid) for pid in place_ids]

    # Tập biến thực sự xuất hiện trong BDD
    support_vars = set(bdd.support)

    # Biến không nằm trong support: BDD không phụ thuộc → "tự do" toàn cục
    free_indices = [i for i, v in enumerate(place_vars) if v not in support_vars]

    free_constant = 0  
    free_values = {}   
    for i in free_indices:
        w = int(c[i])
        if w >= 0:
            free_values[i] = 1
            free_constant += w
        else:
            free_values[i] = 0

    best_marking: Optional[List[int]] = None
    best_value: Optional[int] = None

    # Duyệt tất cả satisfying assignments của BDD
    for point in bdd.satisfy_all():
        value = free_constant
        marking = [0] * num_places

        for i, var in enumerate(place_vars):
            w = int(c[i])

            if var in point:
                val = int(point[var])
                marking[i] = val
                value += w * val

            elif i in free_values:
                val = free_values[i]
                marking[i] = val

            else:
                if w >= 0:
                    val = 1
                    value += w
                else:
                    val = 0
                marking[i] = val

        if best_value is None or value > best_value:
            best_value = value
            best_marking = marking

    return best_marking, best_value




