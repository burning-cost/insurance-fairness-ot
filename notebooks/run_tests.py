# Databricks notebook source
# MAGIC %pip install polars networkx POT scipy statsmodels scikit-learn pytest

# COMMAND ----------

import subprocess, sys, os, tempfile, re

work_dir = tempfile.mkdtemp(prefix="insurance_fairness_ot_")
repo_url = "https://github.com/burning-cost/insurance-fairness-ot.git"
clone = subprocess.run(["git", "clone", repo_url, work_dir], capture_output=True, text=True, timeout=120)
if clone.returncode != 0:
    dbutils.notebook.exit("CLONE_FAILED: " + clone.stderr[-1000:])

# Install package in editable mode
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", work_dir],
    capture_output=True, text=True, timeout=120
)
if install.returncode != 0:
    dbutils.notebook.exit("INSTALL_FAILED: " + install.stderr[-1000:])

result = subprocess.run(
    [sys.executable, "-m", "pytest", os.path.join(work_dir, "tests"), "-v", "--tb=short"],
    capture_output=True, text=True, timeout=300, cwd=work_dir,
)

output = result.stdout + result.stderr
clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)

exit_msg = f"EXIT_CODE:{result.returncode}\n" + clean_output[-6000:]
dbutils.notebook.exit(exit_msg)
