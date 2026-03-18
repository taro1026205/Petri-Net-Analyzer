from collections import deque
import numpy as np
from .PetriNet import PetriNet
from typing import Set, Tuple

def bfs_reachable(pn: PetriNet) -> Set[Tuple[int, ...]]:
    initial_marking = tuple(pn.M0)
    visited = {initial_marking}
    queue = deque([pn.M0])
    
    # Lấy số lượng transition từ shape của ma trận I để tránh lỗi IndexOutOfBounds
    num_trans = pn.I.shape[0]
    
    while queue:
        current_marking = queue.popleft()
        
        for t in range(num_trans):
            input_vec = pn.I[t]
            output_vec = pn.O[t]
            
            # 1. Kiểm tra điều kiện kích hoạt cơ bản: M >= I
            if np.all(current_marking >= input_vec):
                # Tính marking tiếp theo
                next_marking = current_marking - input_vec + output_vec
                
                # 2. QUAN TRỌNG: Kiểm tra ràng buộc 1-safe
                # Nếu bất kỳ vị trí nào có > 1 token, transition này không hợp lệ trong 1-safe net
                if np.any(next_marking > 1):
                    continue
                
                next_marking_tuple = tuple(next_marking)
                
                if next_marking_tuple not in visited:
                    visited.add(next_marking_tuple)
                    queue.append(next_marking)
                    
    return visited