"""
Run insurance-fairness-ot tests on Databricks serverless compute.

Usage:
    uv run python run_tests_databricks.py
"""

import base64
import os
import sys
import time
from pathlib import Path

# Load credentials
env_path = Path.home() / ".config/burning-cost/databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute as _compute
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-fairness-ot"


def upload_file(local_path: Path, remote_path: str) -> None:
    remote_dir = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=remote_dir)
    except Exception:
        pass
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ImportFormat.AUTO,
        overwrite=True,
    )


SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache"}

print("Uploading project files to Databricks workspace...")
for fpath in sorted(PROJECT_ROOT.rglob("*")):
    if not fpath.is_file():
        continue
    if fpath.suffix not in (".py", ".toml", ".md", ".txt"):
        continue
    rel = fpath.relative_to(PROJECT_ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        continue
    remote = f"{WORKSPACE_PATH}/{rel}".replace("\\", "/")
    upload_file(fpath, remote)
    print(f"  Uploaded: {rel}")

print("Upload complete.")

# Create test runner notebook
NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %pip install pytest polars networkx POT scipy statsmodels numpy --quiet

# COMMAND ----------

import subprocess, sys, os, shutil

# Install the package
r_install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-fairness-ot", "--quiet", "--no-deps"],
    capture_output=True, text=True,
)

# Copy project to /tmp to avoid __pycache__ issues on Workspace FS
shutil.copytree("/Workspace/insurance-fairness-ot", "/tmp/insurance-fairness-ot", dirs_exist_ok=True)

# Run the tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/tmp/insurance-fairness-ot/tests",
     "-v", "--tb=short",
     "--rootdir=/tmp/insurance-fairness-ot",
    ],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
         "PYTHONPATH": "/tmp/insurance-fairness-ot/src"},
)
test_out = result.stdout + ("\\nSTDERR:\\n" + result.stderr if result.stderr else "")

summary = (
    f"=== INSTALL (rc={r_install.returncode}) ===\\n{r_install.stderr[-200:] if r_install.stderr else 'ok'}\\n\\n"
    f"=== TESTS (exit={result.returncode}) ===\\n"
    + (test_out[-7000:] if len(test_out) > 7000 else test_out)
)

dbutils.notebook.exit(summary)
"""

test_nb_path = f"{WORKSPACE_PATH}/run_tests_nb"
encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=test_nb_path,
    content=encoded_nb,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Test notebook uploaded to {test_nb_path}")

# Submit with serverless compute
print("Submitting test job (serverless)...")
run_resp = w.jobs.submit(
    run_name="insurance-fairness-ot-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=test_nb_path,
                base_parameters={},
            ),
            environment_key="serverless",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="serverless",
            spec=_compute.Environment(client="2"),
        )
    ],
)

run_id = run_resp.run_id
print(f"Job run ID: {run_id}")

# Poll
print("Waiting for job to complete...")
while True:
    state = w.jobs.get_run(run_id=run_id)
    life = state.state.life_cycle_state.value if state.state.life_cycle_state else "UNKNOWN"
    result_state = state.state.result_state.value if state.state.result_state else None
    print(f"  State: {life} / {result_state}")
    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

# Retrieve notebook output
nb_result = ""
try:
    tasks = w.jobs.get_run(run_id=run_id).tasks or []
    for task in tasks:
        try:
            out = w.jobs.get_run_output(run_id=task.run_id)
            if out.notebook_output and out.notebook_output.result:
                nb_result = out.notebook_output.result
                break
        except Exception as e:
            print(f"  Could not retrieve task output: {e}")
except Exception as e:
    print(f"  Could not retrieve run details: {e}")

print("\n" + "=" * 60)
print(nb_result[-10000:] if len(nb_result) > 10000 else nb_result)
print("=" * 60)

if result_state != "SUCCESS":
    print(f"\nJob FAILED at infrastructure level. State: {result_state}")
    sys.exit(1)

import re
m = re.search(r"exit=(\d+)", nb_result)
test_exit = int(m.group(1)) if m else -1

if test_exit == 0:
    print("\nTests PASSED on Databricks.")
    sys.exit(0)
else:
    print(f"\nTests FAILED (exit code {test_exit}).")
    sys.exit(1)
