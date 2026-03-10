"""
Run insurance-fairness-ot tests on Databricks serverless by cloning from GitHub.
Execute with: uv run python run_databricks_tests.py (from burning-cost project root)
"""
import os
import sys
import time
import json
import base64
import requests

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# ── Notebook script ───────────────────────────────────────────────────────────
NOTEBOOK_SCRIPT = '''
import subprocess, sys, os, shutil, tempfile

repo_dir = tempfile.mkdtemp(prefix="ift_")
try:
    r = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/burning-cost/insurance-fairness-ot.git", repo_dir],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError("Clone failed: " + r.stderr[:500])

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "numpy>=1.24", "scipy>=1.10", "networkx>=3.0",
         "POT>=0.9", "polars>=0.20", "pytest>=7.0"],
        capture_output=True, text=True, check=False
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", repo_dir],
        capture_output=True, text=True, check=False
    )

    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         os.path.join(repo_dir, "tests"), "--tb=short", "-q"],
        capture_output=True, text=True, cwd=repo_dir
    )
    combined = r.stdout + (r.stderr if r.stderr else "")
    # Display last 5000 chars to stay under notebook output limit
    displayHTML("<pre>" + combined[-5000:].replace("<","&lt;").replace(">","&gt;") + "</pre>")
    print(combined[-3000:])
    if r.returncode != 0:
        raise RuntimeError(f"TESTS FAILED rc={r.returncode}\\n" + combined[-2000:])
    print("ALL TESTS PASSED")

finally:
    shutil.rmtree(repo_dir, ignore_errors=True)
'''

# ── Upload script to workspace ────────────────────────────────────────────────
script_b64 = base64.b64encode(NOTEBOOK_SCRIPT.encode()).decode()
workspace_path = "/Workspace/insurance-fairness-ot/test_runner"

requests.post(
    f"{HOST}/api/2.0/workspace/mkdirs",
    headers=HEADERS,
    json={"path": "/Workspace/insurance-fairness-ot"},
)

resp = requests.post(
    f"{HOST}/api/2.0/workspace/import",
    headers=HEADERS,
    json={
        "path": workspace_path,
        "format": "SOURCE",
        "language": "PYTHON",
        "content": script_b64,
        "overwrite": True,
    }
)
print(f"Workspace import: {resp.status_code}", "OK" if resp.status_code == 200 else resp.text[:200])

# ── Submit serverless job ─────────────────────────────────────────────────────
run_resp = requests.post(
    f"{HOST}/api/2.1/jobs/runs/submit",
    headers=HEADERS,
    json={
        "run_name": "insurance-fairness-ot-tests",
        "tasks": [{"task_key": "run-tests", "notebook_task": {"notebook_path": workspace_path}}],
    },
)
print(f"Submit: {run_resp.status_code}")
run_data = run_resp.json()
if "run_id" not in run_data:
    print("Failed:", json.dumps(run_data, indent=2))
    sys.exit(1)

run_id = run_data["run_id"]
print(f"Run ID: {run_id}")

# ── Poll for completion ───────────────────────────────────────────────────────
print("\nPolling for completion...")
for i in range(90):
    time.sleep(20)
    status_resp = requests.get(
        f"{HOST}/api/2.1/jobs/runs/get",
        headers=HEADERS,
        params={"run_id": run_id},
    )
    run_info = status_resp.json()
    state = run_info.get("state", {})
    life_cycle = state.get("life_cycle_state", "UNKNOWN")
    result_state = state.get("result_state", "")
    print(f"  [{i*20}s] {life_cycle} {result_state}")

    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        print(f"\nFinal: {life_cycle} / {result_state}")

        tasks = run_info.get("tasks", [])
        for task in tasks:
            task_run_id = task.get("run_id")
            if not task_run_id:
                continue
            output_resp = requests.get(
                f"{HOST}/api/2.1/jobs/runs/get-output",
                headers=HEADERS,
                params={"run_id": task_run_id},
            )
            out = output_resp.json()
            nb = out.get("notebook_output", {})
            if nb.get("result"):
                print("\n=== OUTPUT ===")
                print(nb["result"])
            err = out.get("error", "")
            if err:
                print("\n=== ERROR ===")
                print(err[:3000])
            trace = out.get("error_trace", "")
            if trace:
                print("\n=== TRACE ===")
                print(trace[-2000:])

        if result_state == "SUCCESS":
            print("\nAll tests PASSED.")
            sys.exit(0)
        else:
            print(f"\nFAILED: {result_state}")
            sys.exit(1)

print("Timed out")
sys.exit(1)
