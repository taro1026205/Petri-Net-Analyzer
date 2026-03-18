import numpy as np
from .PetriNet import PetriNet
from typing import Set, Tuple

def dfs_reachable(pn: PetriNet) -> Set[Tuple[int, ...]]:
    initial_marking = tuple(pn.M0)
    visited = {initial_marking}
    stack = [pn.M0]
    
    # Lấy số lượng transition từ shape của ma trận I
    num_trans = pn.I.shape[0]
    
    while stack:
        current_marking = stack.pop()
        
        for t in range(num_trans):
            input_vec = pn.I[t]
            output_vec = pn.O[t]
            
            # 1. Kiểm tra điều kiện kích hoạt cơ bản
            if np.all(current_marking >= input_vec):
                next_marking = current_marking - input_vec + output_vec
                
                # 2. Kiểm tra ràng buộc 1-safe
                if np.any(next_marking > 1):
                    continue

                next_marking_tuple = tuple(next_marking)
                
                if next_marking_tuple not in visited:
                    visited.add(next_marking_tuple)
                    stack.append(next_marking)
    
    return visited