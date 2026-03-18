# Petri Net Analyzer — CO2011 Assignment

A Python toolkit for symbolic and algebraic reasoning on **1-safe Petri nets**, combining BFS/DFS explicit traversal, BDD-based symbolic reachability, ILP-based deadlock detection, and linear optimization over reachable markings.

## Features

| Module | Description |
|--------|-------------|
| `PetriNet.py` | PNML parser — builds place/transition matrices (I, O) and initial marking M0 |
| `BFS.py` | Explicit BFS reachability with 1-safe enforcement |
| `DFS.py` | Explicit DFS reachability with 1-safe enforcement |
| `BDD.py` | Symbolic reachability via iterative image computation using PyEDA BDDs |
| `Deadlock.py` | ILP + BDD hybrid deadlock detection (PuLP + CBC solver) |
| `Optimization.py` | Maximize `c⊤M` over all reachable markings using the BDD |

## Project Structure

```
.
├── main.py                  # Entry point — runs all tasks in sequence
├── petri_net/
│   ├── PetriNet.py
│   ├── BFS.py
│   ├── DFS.py
│   ├── BDD.py
│   ├── Deadlock.py
│   └── Optimization.py
├── input/
│   └── example.pnml         # Sample 1-safe Petri net (PNML format)
├── input.txt                # Weight vectors c for optimization (one per line)
└── requirements.txt
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tasks on example.pnml
python main.py
```

To test a different net, edit the `filename` variable in `main.py`.

## Optimization Input Format

`input.txt` — one weight vector per line (space-separated integers, length = number of places):

```
1 0 1 0 1
0 1 0 1 0
```

## Dependencies

- `pyeda==0.28.0` — BDD construction and symbolic operations
- `pulp>=2.6` — ILP solver (uses CBC backend)
- `numpy>=1.24` — matrix operations

## Course

**CO2011 — Mathematical Modeling**  
Ho Chi Minh City University of Technology (HCMUT), Semester 1 — 2025/2026

---

## Group Members

| Student ID | Full Name |
| :--- | :--- |
| **2352082** | HOÀNG XUÂN BÁCH |
| **2352424** | NGUYỄN VIỆT HÙNG |
| **2352520** | LÊ ĐIỂN VINH KHÁNH |
| **2352824** | HỒ HỒNG PHÚC NGUYÊN |
| **2353283** | NGUYỄN TRẦN TRỌNG TUYÊN |

---
