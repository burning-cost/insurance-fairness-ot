"""
Upload the package to Databricks and run tests via the Jobs API.
"""
import os
import sys
import time
import json
import base64

# Load credentials
with open(os.path.expanduser("~/.config/burning-cost/databricks.env")) as f:
    for line in f:
        line = line.strip()
        if line and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

# ── 1. Upload package source ──────────────────────────────────────────────────
print("Uploading source files to Databricks workspace...")

import pathlib
import zipfile
import io

repo_root = pathlib.Path("/home/ralph/repos/insurance-fairness-ot")

# Create zip of src + tests
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in repo_root.rglob("*.py"):
        if "run_databricks_tests" in str(p) or "run_tests_databricks" in str(p):
            continue
        arcname = p.relative_to(repo_root)
        zf.write(p, arcname)
    # Also include pyproject.toml
    zf.write(repo_root / "pyproject.toml", "pyproject.toml")

buf.seek(0)
zip_bytes = buf.read()

# Upload zip to DBFS (more reliable than workspace for binary)
dbfs_path = "dbfs:/tmp/insurance-fairness-ot/package.zip"
w.dbfs.upload(dbfs_path, zip_bytes, overwrite=True)
print(f"Uploaded {len(zip_bytes):,} bytes to {dbfs_path}")

# ── 2. Create test notebook ───────────────────────────────────────────────────
notebook_content = b'''# Databricks notebook source
# insurance-fairness-ot test runner

# COMMAND ----------
import subprocess, sys, os, zipfile, pathlib, tempfile

# COMMAND ----------
# Extract package from DBFS
dbfs_zip = "/dbfs/tmp/insurance-fairness-ot/package.zip"
extract_dir = "/tmp/insurance-fairness-ot"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(dbfs_zip) as zf:
    zf.extractall(extract_dir)
print("Extracted to", extract_dir)
import subprocess, sys
print("Python:", sys.executable)

# COMMAND ----------
# Install dependencies
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "numpy>=1.24", "scipy>=1.10", "statsmodels>=0.14",
     "networkx>=3.0", "POT>=0.9", "polars>=0.20",
     "pytest>=7.0"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
else:
    print("Dependencies installed OK")

# COMMAND ----------
# Install package in editable mode
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", extract_dir],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
else:
    print("Package installed OK")

# COMMAND ----------
# Run tests (full suite, no -x so we see all failures)
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(extract_dir, "tests"),
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    cwd=extract_dir
)
print(result.stdout[-8000:])
if result.stderr:
    print("STDERR:", result.stderr[-500:])
print("Return code:", result.returncode)
if result.returncode != 0:
    raise Exception(f"Tests FAILED (return code {result.returncode})")
else:
    print("ALL TESTS PASSED")
'''

# Upload notebook to workspace
workspace_dir = "/Workspace/insurance-fairness-ot"
try:
    w.workspace.mkdirs(workspace_dir)
except Exception:
    pass

notebook_path = f"{workspace_dir}/run_tests"

import base64
w.workspace.import_(
    path=notebook_path,
    content=base64.b64encode(notebook_content).decode(),
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Uploaded notebook to {notebook_path}")

# ── 3. Submit job ─────────────────────────────────────────────────────────────
print("Submitting test job...")

run_waiter = w.jobs.submit(
    run_name="insurance-fairness-ot tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
            ),
            new_cluster=compute.ClusterSpec(
                spark_version="15.4.x-scala2.12",
                node_type_id="m5d.large",
                num_workers=0,
                spark_conf={"spark.master": "local[*]"},
            ),
        )
    ],
)

run_id = run_waiter.run_id
print(f"Job run ID: {run_id}")

# ── 4. Poll for completion ────────────────────────────────────────────────────
print("Waiting for job to complete (polling every 20s)...")

while True:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    life_cycle = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    result_str = f" / {state.result_state.value}" if state.result_state else ""
    print(f"  [{time.strftime('%H:%M:%S')}] State: {life_cycle}{result_str}")

    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

# ── 5. Get output ─────────────────────────────────────────────────────────────
final_run = w.jobs.get_run(run_id=run_id)
result_state = final_run.state.result_state.value if final_run.state.result_state else "UNKNOWN"
print(f"\nFinal result: {result_state}")

tasks = final_run.tasks or []
for task in tasks:
    task_run_id = task.run_id
    if task_run_id is None:
        continue
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output and output.notebook_output.result:
            print("\n=== NOTEBOOK OUTPUT ===")
            print(output.notebook_output.result[:8000])
        if output.error:
            print("\n=== ERROR ===")
            print(output.error[:2000])
        if output.error_trace:
            print("\n=== TRACE ===")
            print(output.error_trace[:2000])
    except Exception as e:
        print(f"Could not get task output: {e}")

if result_state == "SUCCESS":
    print("\nAll tests PASSED.")
    sys.exit(0)
else:
    print(f"\nJob FAILED with result: {result_state}")
    sys.exit(1)
