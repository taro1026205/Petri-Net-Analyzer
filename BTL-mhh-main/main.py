from petri_net.PetriNet import PetriNet
from petri_net.BDD import bdd_reachable
from petri_net.Optimization import max_reachable_marking
from petri_net.BFS import bfs_reachable
from petri_net.DFS import dfs_reachable
from petri_net.Deadlock import find_deadlock_ilp_bdd
from pyeda.inter import * 
import numpy as np
import time         # estimate time
## from graphviz import Source

def main():
    # 1. Load Petri Net từ file PNML
    # ------------------------------------------------------
    filename = "input/example.pnml"   # đổi file tại đây
    print("Loading PNML:", filename)

    pn = PetriNet.from_pnml(filename)
    print("\n--- Petri Net Loaded ---")
    print(pn)

    # ------------------------------------------------------
    # 2. BFS reachable
    # ------------------------------------------------------
    print("\n--- BFS Reachable Markings ---")
    start_bfs = time.time()
    bfs_set = bfs_reachable(pn)
    end_bfs = time.time()
    # for m in bfs_set:
    #     print(np.array(m))
    print("Total BFS reachable =", len(bfs_set))
    print(f"BFS Time taken: {end_bfs - start_bfs:.4f} seconds")

    # # ------------------------------------------------------
    # # 3. DFS reachable
    # # ------------------------------------------------------
    print("\n--- DFS Reachable Markings ---")
    start_dfs = time.time()
    dfs_set = dfs_reachable(pn)
    end_dfs = time.time()
    # for m in dfs_set:
    #     print(np.array(m))
    print("Total DFS reachable =", len(dfs_set))
    print(f"DFS Time taken: {end_dfs - start_dfs:.4f} seconds")

    # # ------------------------------------------------------
    # # 4. BDD reachable
    # # ------------------------------------------------------
    print("\n--- BDD Reachable ---")
    start_bdd = time.time()
    bdd, count = bdd_reachable(pn)
    end_bdd = time.time()
    
    print("--- Satisfying assignments ---")
    for sat in bdd.satisfy_all():
        line = ", ".join(f"{var.names[0]}={val}" for var, val in sat.items())
        print(line)
    print("BDD reachable markings =", count)
    ## Source(bdd.to_dot()).render("bdd", format="png", cleanup=True)
    print(f"BDD Time taken: {end_bdd - start_bdd:.4f} seconds")


    # # ------------------------------------------------------
    # # 5. Deadlock detection
    # # ------------------------------------------------------
    print("\n--- Deadlock reachable marking ---")
    start_deadlock = time.time()
    dead = find_deadlock_ilp_bdd(pn, bdd)
    end_deadlock = time.time()
    print(f"Deadlock detection Time taken: {end_deadlock - start_deadlock:.4f} seconds")

    if dead is not None:
        print("Deadlock marking:", dead)
    else:
        print("No deadlock reachable.")

    # # # ------------------------------------------------------
    # # # 6. Optimization: maximize c·M
    # # # ------------------------------------------------------
    print("\n--- Optimize c·M ---")
    with open("input.txt", "r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            C = np.array(list(map(int, line.split())))
            max_mark, max_val = max_reachable_marking(pn.place_ids, bdd, C)
            print("Max marking:", " ".join(map(str, max_mark)), "Max value:", max_val)


if __name__ == "__main__":
    main()
