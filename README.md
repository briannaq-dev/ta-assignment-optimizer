# TA Assignment Optimizer
### Project for DS3500: Advanced Programming with Data SEC 01 F2024

Assigns teaching assistants to lab sections using a **multi-objective evolutionary algorithm** written from scratch in Python.

Given 17 sections and 43 TAs, each with a preference per section (**P**referred, **W**illing, **U**nwilling), the optimizer searches for assignments that trade off five goals at once:

| Objective | Minimizes |
|-----------|-----------|
| `overallocation` | TAs assigned more sections than they agreed to |
| `conflicts` | TAs scheduled in two sections at the same time |
| `undersupport` | Sections with fewer TAs than their minimum |
| `unwilling` | Assignments to sections a TA is unwilling to support |
| `unpreferred` | Assignments to sections a TA is only willing (not preferring) to support |

## How it works
- `evo.py` is a small evolutionary framework: an `Environment` holds a population of solutions, registered objective functions and "agents" (operators that tweak a solution). Each round it applies a random agent to random solutions, then periodically removes dominated solutions so only the **Pareto front** survives.
- `assignta.py` defines the problem: the five objectives, four agents (greedy assignment, unwillingness reduction, time-conflict resolution and random perturbation), and a weighted scoring step that picks one "best" solution from the Pareto front.
- `profiler.py` is a decorator-based profiler. `profiler_report.txt` shows where the time goes (about 100,000 agent calls and 3.4 million dominance checks in a 5-minute run).

## Results
The best solution found has no overallocation, no time conflicts and no unwilling assignments, with one section short a TA and five assignments to merely "willing" TAs (see `best_solution.txt`). The full Pareto front is in `pareto_front.csv`.

## Running
```
pip install numpy pandas pytest
python assignta.py      # evolves for up to 5 minutes, writes pareto_front.csv, best_solution.txt, profiler_report.txt
pytest test_assignta.py # unit tests for each objective, using test1-3.csv
```

TA names are replaced with IDs in the data files.
