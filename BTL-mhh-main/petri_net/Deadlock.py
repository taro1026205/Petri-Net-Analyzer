from typing import Optional, Tuple, List
import numpy as np

import pulp
from pyeda.inter import bddvar

from .PetriNet import PetriNet
from .BDD import bdd_reachable


def _build_deadlock_ilp(pn: PetriNet) -> Tuple["pulp.LpProblem", List["pulp.LpVariable"]]:

    num_trans = pn.I.shape[0]

    # Mô hình ILP chỉ cần tìm nghiệm khả thi ⇒ Minimize 0
    prob = pulp.LpProblem("Deadlock_Search", pulp.LpMinimize)
    x_vars: List[pulp.LpVariable] = []

    # Biến nhị phân cho từng place
    for pid in pn.place_ids:
        x = pulp.LpVariable(f"x_{pid}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        x_vars.append(x)

    # Dummy objective = 0
    prob += 0, "dummy_objective"

    # Ràng buộc 'not enabled' cho từng transition
    for t_idx in range(num_trans):
        inputs_idx = np.where(pn.I[t_idx] == 1)[0]
        outputs_idx = np.where(pn.O[t_idx] == 1)[0]

        # Pure outputs: những place có token mới xuất hiện sau khi bắn
        pure_outputs_idx = [i for i in outputs_idx if i not in inputs_idx]

        terms = []

        # Nếu một input = 0, transition bị chặn
        for i in inputs_idx:
            terms.append(1 - x_vars[i])

        # Nếu một pure_output đã = 1, transition cũng bị chặn (vi phạm 1-safe)
        for i in pure_outputs_idx:
            terms.append(x_vars[i])

        if not terms:
            # Trường hợp hiếm: transition không có input và không có pure_output
            # => luôn enabled trong mọi marking ⇒ không thể có deadlock toàn cục.
            # Ta thêm ràng buộc 0 >= 1 để làm mô hình vô nghiệm.
            prob += 0 >= 1, f"no_deadlock_due_to_transition_{t_idx}"
        else:
            prob += pulp.lpSum(terms) >= 1, f"t_{t_idx}_not_enabled"

    return prob, x_vars


def _is_marking_reachable(reached_bdd, pn: PetriNet, marking: np.ndarray) -> bool:
    assignment = {bddvar(p_id): int(marking[i]) for i, p_id in enumerate(pn.place_ids)}
    restricted = reached_bdd.restrict(assignment)
    return restricted.is_one()


def find_deadlock_ilp_bdd(
    pn: PetriNet,
    time_limit: Optional[int] = None,
    max_iterations: int = 1000,
) -> Tuple[Optional[np.ndarray], int]:
  
    # 1. Symbolic reachable set bằng BDD
    reached_bdd, _ = bdd_reachable(pn)

    # 2. Xây mô hình ILP cho dead marking
    prob, x_vars = _build_deadlock_ilp(pn)

    # 3. Cấu hình solver
    if time_limit is not None:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        status = prob.solve(solver)
        status_str = pulp.LpStatus.get(prob.status, "Unknown")

        if status_str != "Optimal":
            # Mô hình vô nghiệm (Infeasible) hoặc lỗi khác ⇒ không có deadlock
            return None, iterations

        # Đọc nghiệm ILP ra thành marking 0/1
        marking = np.array(
            [int(round(x.varValue)) for x in x_vars],
            dtype=int,
        )

        # 4. Kiểm tra reachable bằng BDD
        if _is_marking_reachable(reached_bdd, pn, marking):
            # Đây là 1 reachable deadlock
            return marking, iterations

        # 5. Nếu marking này không reachable, ta thêm "cut" loại bỏ nó khỏi ILP:
        #    sum_{i: m_i=1} (1 - x_i) + sum_{i: m_i=0} x_i ≥ 1
        #    ⇔ nghiệm mới phải khác ít nhất 1 bit so với marking hiện tại.
        cut_terms = []
        for i, val in enumerate(marking):
            if val == 1:
                cut_terms.append(1 - x_vars[i])
            else:
                cut_terms.append(x_vars[i])

        prob += pulp.lpSum(cut_terms) >= 1, f"cut_{iterations}"

    # Hết số vòng lặp cho phép mà không tìm được reachable deadlock
    return None, iterations


def pretty_print_marking(pn: PetriNet, marking: np.ndarray) -> str:
    parts = []
    for pid, tokens in zip(pn.place_ids, marking):
        parts.append(f"{pid}:{int(tokens)}")
    return ", ".join(parts)
