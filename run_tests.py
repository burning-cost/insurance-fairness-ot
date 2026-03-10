"""Submit and monitor test run on Databricks."""
import os
import time
import sys
import base64
import requests

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
host = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {"Authorization": f"Bearer {token}"}

# Upload the notebook
NOTEBOOK_LOCAL = os.path.join(os.path.dirname(__file__), "notebooks", "run_tests.py")
NOTEBOOK_PATH = "/Workspace/insurance-fairness-ot-tests/run_tests"

with open(NOTEBOOK_LOCAL, "rb") as f:
    content_b64 = base64.b64encode(f.read()).decode()

w.workspace.mkdirs("/Workspace/insurance-fairness-ot-tests")
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=content_b64,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Uploaded {NOTEBOOK_PATH}")

# Check available node types to pick a valid one
r = requests.get(f"{host}/api/2.0/clusters/list-node-types", headers=headers)
node_types = [n["node_type_id"] for n in r.json().get("node_types", [])]
# Pick a small node
preferred = ["i3.xlarge", "m5d.large", "m5.large", "Standard_DS3_v2"]
node_type = next((n for n in preferred if n in node_types), node_types[0] if node_types else "i3.xlarge")
print(f"Using node type: {node_type}")

# Get a valid Spark version
r2 = requests.get(f"{host}/api/2.0/clusters/spark-versions", headers=headers)
versions = [v["key"] for v in r2.json().get("versions", [])]
# Prefer ml runtime with Python 3.11+
preferred_versions = [v for v in versions if "15.4.x-cpu-ml" in v or "15.3.x-cpu-ml" in v or "14.3.x-cpu-ml" in v]
spark_ver = preferred_versions[0] if preferred_versions else "15.4.x-cpu-ml-scala2.12"
print(f"Using Spark version: {spark_ver}")

payload = {
    "run_name": "insurance-fairness-ot-tests",
    "tasks": [{
        "task_key": "run_tests",
        "notebook_task": {"notebook_path": NOTEBOOK_PATH},
        "new_cluster": {
            "spark_version": spark_ver,
            "node_type_id": node_type,
            "num_workers": 0,
            "spark_conf": {"spark.master": "local[*, 4]"},
        },
    }],
}

resp = requests.post(f"{host}/api/2.1/jobs/runs/submit", json=payload, headers=headers)
if not resp.ok:
    print(f"Error: {resp.status_code} {resp.text}")
    sys.exit(1)
run_id = resp.json()["run_id"]
print(f"Submitted run_id={run_id}")

# Poll
while True:
    r = requests.get(f"{host}/api/2.1/jobs/runs/get?run_id={run_id}", headers=headers)
    r.raise_for_status()
    data = r.json()
    state = data["state"]["life_cycle_state"]
    print(f"  {state}")
    if state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

result_state = data["state"].get("result_state", "UNKNOWN")
print(f"\nFinal: {result_state}")

# Get output from each task
for task in data.get("tasks", []):
    task_run_id = task["run_id"]
    out = requests.get(
        f"{host}/api/2.1/jobs/runs/get-output?run_id={task_run_id}",
        headers=headers
    )
    out.raise_for_status()
    out_data = out.json()
    if "notebook_output" in out_data:
        print("\n=== NOTEBOOK OUTPUT ===")
        print(out_data["notebook_output"].get("result", ""))
    if "error" in out_data and out_data["error"]:
        print(f"\nERROR: {out_data['error']}")
    if "error_trace" in out_data and out_data["error_trace"]:
        print(out_data["error_trace"][:4000])

sys.exit(0 if result_state == "SUCCESS" else 1)
