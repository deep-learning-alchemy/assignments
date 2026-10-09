"""Run Problem 4 measurements and continuations on Modal H100s (or locally) and collect them.

A job is a dict with a unique `name` and either
    {"measure": {"run": ..., "step": ..., **measure options}}          -> <name>.json
    {"continue": {"run": ..., "step": ..., "transform": ..., ...}}      -> <name>.jsonl
Results are written to results/a3-inside-the-hessian/ on your Modal volume, and fetch()
copies them to RESULTS_DIR. Jobs whose result already exists on the volume are skipped,
so re-running a launcher only launches what is missing.

    launch(jobs)               # spawn on Modal, detached; returns immediately
    launch(jobs, local=True)   # run here, one after another (needs a CUDA GPU)
    fetch(names)               # copy finished results from your volume into RESULTS_DIR
    load_result(name)          # read a fetched result
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import modal
from modal.exception import NotFoundError

from modal_utils import (
    MODAL_DATA_DIR,
    MODAL_ENVIRONMENT,
    VOLUME_MOUNTS,
    build_image,
    timestamped_modal_app_name,
    user_volume,
)

EXPERIMENT_KEY = "a3-inside-the-hessian"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / EXPERIMENT_KEY
VOLUME_RESULTS = f"results/{EXPERIMENT_KEY}"  # relative to the root of your Modal volume


def _filename(job):
    return job["name"] + (".json" if "measure" in job else ".jsonl")


def run_job(job, results_dir):
    """Execute one job and write its result into results_dir (atomically)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / _filename(job)
    if path.exists():
        return str(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    if "measure" in job:
        from experiments.a3_optimization.hessian_measure import measure_checkpoint

        tmp.write_text(json.dumps(measure_checkpoint(**job["measure"])))
    elif "continue" in job:
        from experiments.a3_optimization.hessian_measure import continue_projected

        continue_projected(**job["continue"], log=tmp)
    else:
        raise ValueError(f"Job {job['name']!r} needs a 'measure' or 'continue' entry.")
    tmp.rename(path)
    return str(path)


def _check(jobs):
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("Job names must be unique.")
    for j in jobs:
        spec = j.get("continue")
        if spec and not isinstance(spec.get("transform", "full"), str):
            raise ValueError("On Modal, pass a transform by name (see hessian_measure.ARMS); "
                             "add your own transform to ARMS to run it remotely.")


def _volume_results():
    try:
        return {PurePosixPath(e.path).name for e in user_volume.listdir(VOLUME_RESULTS)}
    except NotFoundError:  # nothing has finished yet
        return set()


app = modal.App("dl-alchemy-a3-hessian")


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=6 * 60 * 60)
def _run_job_remote(job):
    try:
        return run_job(job, MODAL_DATA_DIR / VOLUME_RESULTS)
    finally:
        user_volume.commit()


def launch(jobs, local=False, max_parallel=8):
    """Run jobs on Modal (detached, at most max_parallel at once) or here."""
    _check(jobs)
    if local:
        return [run_job(j, RESULTS_DIR) for j in jobs]
    done = _volume_results()
    pending = [j for j in jobs if _filename(j) not in done]
    for j in jobs:
        print(("launch " if j in pending else "done   ") + j["name"])
    if not pending:
        return []
    with modal.enable_output(), app.run(
        name=timestamped_modal_app_name("dl-alchemy-a3-hessian"),
        environment_name=MODAL_ENVIRONMENT,
        detach=True,
    ):
        remote = _run_job_remote.with_options(max_containers=max_parallel)
        calls = [remote.spawn(j) for j in pending]
    return [c.object_id for c in calls]


def fetch(names=None):
    """Copy finished results from your Modal volume into RESULTS_DIR (all if names is None)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for filename in sorted(_volume_results()):
        stem = filename.rsplit(".", 1)[0]
        if filename.endswith(".partial") or (names is not None and stem not in names):
            continue
        dest = RESULTS_DIR / filename
        if not dest.exists():
            tmp = dest.with_name(filename + ".partial")
            with open(tmp, "wb") as f:
                for chunk in user_volume.read_file(f"{VOLUME_RESULTS}/{filename}"):
                    f.write(chunk)
            tmp.rename(dest)
        got.append(stem)
    return got


def load_result(name):
    """A fetched result: a dict for measurements, a list of records for continuations."""
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    path = RESULTS_DIR / f"{name}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    raise FileNotFoundError(f"No result {name!r} in {RESULTS_DIR}; run fetch() once the job finishes.")
