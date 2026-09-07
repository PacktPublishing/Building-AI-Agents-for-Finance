"""Score agent frameworks with the Chapter 3 decision rubric.

Edit decision_matrix.csv first: replace the illustrative 0-5 scores with
your own pilot results and adjust the weights to your priorities (weights
must sum to 1.0). Then run:

    python score_frameworks.py [path/to/decision_matrix.csv]

The script prints each framework's weighted score (0-5 scale) in rank
order, plus the dimension that most helped and most hurt it.
"""
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "decision_matrix.csv"
rows = list(csv.DictReader(path.open()))

meta = {"dimension", "name", "weight"}
frameworks = [c for c in rows[0] if c not in meta]
weights = {r["dimension"]: float(r["weight"]) for r in rows}
total_w = sum(weights.values())
if abs(total_w - 1.0) > 1e-6:
    print(f"warning: weights sum to {total_w:.2f}, not 1.0 — scores rescaled\n")

results = []
for fw in frameworks:
    contrib = {r["dimension"]: float(r[fw]) * float(r["weight"]) / total_w for r in rows}
    score = sum(contrib.values())
    best = max(contrib, key=contrib.get)
    worst = min(contrib, key=contrib.get)
    results.append((score, fw, best, worst))

print(f"{'rank':<5}{'framework':<24}{'score':<8}{'strongest':<11}weakest")
for i, (score, fw, best, worst) in enumerate(sorted(results, reverse=True), 1):
    print(f"{i:<5}{fw:<24}{score:<8.2f}{best:<11}{worst}")

print("\nShortlist the top 2-3, then let a pilot decide (see the "
      "'Evaluation method at a glance' section, step 4, in Chapter 3).")
