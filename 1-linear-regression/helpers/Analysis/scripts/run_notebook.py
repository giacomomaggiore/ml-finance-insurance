"""Run the notebook from top to bottom in a fresh kernel ("Restart & Run All") and check it.

Why this script: `jupyter nbconvert` from miniconda fails on this Mac (a config entry points
to a missing extension), so notebooks are executed with nbclient directly.

Run with miniconda's python (it has nbformat and nbclient), from anywhere:
    ~/miniconda3/bin/python "helpers/Analysis/scripts/run_notebook.py"            # check only: runs a copy
    ~/miniconda3/bin/python "helpers/Analysis/scripts/run_notebook.py" --save     # run and SAVE the outputs
                                                                          # into the notebook (for submission)
Options: --kernel NAME (default: jupyter-env = the "Python 3.13 (venv)" kernel)

It reports: errors, warnings printed to stderr, slow cells, and what still blocks the
submission (PREP NOTE lines, answer placeholders, missing group names).
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "1-main.ipynb"
kernel = sys.argv[sys.argv.index("--kernel") + 1] if "--kernel" in sys.argv else "jupyter-env"
save = "--save" in sys.argv

nb = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name=kernel, resources={"metadata": {"path": str(ROOT)}},
                        record_timing=True)
t0 = time.time()
failed = None
try:
    client.execute()
except CellExecutionError as err:
    failed = err
total = time.time() - t0

code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
print(f"kernel: {kernel} | total run time: {total:.0f} s | code cells run: "
      f"{sum(1 for _, c in code_cells if c.get('execution_count'))} of {len(code_cells)}")


def seconds(cell):
    t = cell.get("metadata", {}).get("execution", {})
    try:
        from datetime import datetime
        start = datetime.fromisoformat(t["iopub.execute_input"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(t["shell.execute_reply"].replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except (KeyError, ValueError):
        return 0.0


for i, c in code_cells:
    first = c.source.splitlines()[0][:70] if c.source else ""
    for out in c.get("outputs", []):
        if out.get("output_type") == "error":
            print(f"  ERROR in cell {i} ({first}): {out.get('ename')}: {out.get('evalue')}")
        if out.get("output_type") == "stream" and out.get("name") == "stderr":
            print(f"  stderr in cell {i} ({first}): {out.get('text', '').strip()[:200]}")
    if seconds(c) > 20:
        print(f"  slow cell {i} ({first}): {seconds(c):.0f} s")

text = "\n".join(c.source for c in nb.cells)
blockers = {
    "PREP NOTE lines": text.count("PREP NOTE"),
    "answer placeholders ('To be written')": text.count("To be written"),
    "group-name placeholders": text.count("*(name "),
}
print("still to do before submission:", {k: v for k, v in blockers.items() if v} or "nothing")

if save and failed is None:
    nbformat.write(nb, NOTEBOOK)
    print(f"outputs saved into {NOTEBOOK.name}")
elif save:
    print("NOT saved, because a cell failed.")
sys.exit(1 if failed else 0)
