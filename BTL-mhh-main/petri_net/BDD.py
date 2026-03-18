import collections
from typing import Tuple, List, Optional
from pyeda.inter import *
from .PetriNet import PetriNet
from collections import deque
import numpy as np

def bdd_reachable(pn: PetriNet) -> Tuple[BinaryDecisionDiagram, int]:
    """
    Tính toán tập reachable markings sử dụng BDD cho mạng Petri 1-safe.
    """
    # 1. Tạo các hằng số BDD thủ công từ biểu thức (Expression)
    # expr(1) tạo biểu thức True, expr2bdd chuyển nó thành BDD One
    BDD_ONE = expr2bdd(expr(1))
    BDD_ZERO = expr2bdd(expr(0))

    # 2. Tạo biến BDD cho từng Place
    # bddvar tạo biến BDD
    place_vars = [bddvar(p_id) for p_id in pn.place_ids]
    num_places = len(place_vars)
    
    # 3. Mã hóa Marking ban đầu (M0)
    reached = BDD_ONE
    for i, val in enumerate(pn.M0):
        if val == 1:
            reached = reached & place_vars[i]
        else:
            reached = reached & ~place_vars[i]
            
    frontier = reached # Tập biên để duyệt BFS
    
    # 4. Chuẩn bị Logic cho các Transition (Pre-calculation)
    transitions_logic = []
    num_trans = pn.I.shape[0]
    
    for t_idx in range(num_trans):
        inputs_idx = np.where(pn.I[t_idx] == 1)[0]
        outputs_idx = np.where(pn.O[t_idx] == 1)[0]
        
        set_inputs = set(inputs_idx)
        set_outputs = set(outputs_idx)
        
        # pure_inputs: Place mất token (Input mà không phải Output) -> sẽ chuyển thành 0
        pure_inputs_idx = list(set_inputs - set_outputs)
        # pure_outputs: Place nhận token (Output mà không phải Input) -> sẽ chuyển thành 1
        pure_outputs_idx = list(set_outputs - set_inputs)
        
        # --- Xây dựng Guard Condition (Điều kiện kích hoạt) ---
        guard = BDD_ONE
        # Inputs phải có token
        for i in inputs_idx:
            guard = guard & place_vars[i]
        # Safety: Output places phải RỖNG (để đảm bảo 1-safe)
        for i in pure_outputs_idx:
            guard = guard & ~place_vars[i]
        
        # --- Xây dựng Update Logic (Thay đổi trạng thái) ---
        # Danh sách các biến sẽ thay đổi giá trị
        update_vars = []
        for i in pure_inputs_idx:
            update_vars.append(place_vars[i])
        for i in pure_outputs_idx:
            update_vars.append(place_vars[i])
            
        # Mặt nạ giá trị mới: Inputs -> 0, Outputs -> 1
        new_values_mask = BDD_ONE
        for i in pure_inputs_idx:
            new_values_mask = new_values_mask & ~place_vars[i]
        for i in pure_outputs_idx:
            new_values_mask = new_values_mask & place_vars[i]
            
        transitions_logic.append((guard, update_vars, new_values_mask))
        
    # 5. Vòng lặp Symbolic BFS
    while True:
        accumulated_next = BDD_ZERO # Khởi tạo tập rỗng
        
        for guard, update_vars, new_values_mask in transitions_logic:
            # Bước A: Lọc các trạng thái thỏa mãn điều kiện bắn
            fireable_states = frontier & guard
            
            if fireable_states.is_zero():
                continue
                
            # Bước B: "Quên" giá trị cũ của các biến thay đổi (Existential Abstraction / Smoothing)
            # Tương đương với: exists v1, v2... . fireable_states
            smoothed_states = fireable_states.smoothing(update_vars)
            
            # Bước C: Áp đặt giá trị mới (AND với mask giá trị mới)
            next_states = smoothed_states & new_values_mask
            
            # Hợp nhất vào tập trạng thái tiếp theo
            accumulated_next = accumulated_next | next_states
        
        # Nếu không sinh ra được trạng thái nào nữa
        if accumulated_next.is_zero():
            break
            
        # Chỉ lấy các trạng thái mới chưa từng duyệt: New = Next AND (NOT Reached)
        new_frontier = accumulated_next & ~reached
        
        if new_frontier.is_zero():
            break
            
        reached = reached | new_frontier
        frontier = new_frontier

    # 6. Tính số lượng trạng thái
    # satisfy_count chỉ đếm trên các biến có trong support set của BDD.
    # Ta cần nhân thêm 2^(số biến vắng mặt) để ra kết quả chính xác.
    count = reached.satisfy_count()
    support_len = len(reached.support)
    
    if support_len < num_places:
        count = count * (1 << (num_places - support_len))
        
    return reached, count