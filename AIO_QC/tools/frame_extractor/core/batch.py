# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Shared batch planning + a blocking runner (used by the CLI; the GUI polls the
same messages with a QTimer)."""
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import cv2

from AIO_QC.tools.frame_extractor.core.extract import job_process
from AIO_QC.tools.frame_extractor.core.sources import parse_time


def _quick_duration(job):
    """Cheap duration estimate (container metadata) used only for splitting."""
    cap = cv2.VideoCapture(job.video_path, cv2.CAP_FFMPEG)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0.0
    finally:
        cap.release()
    start = parse_time(job.start) if job.start else 0.0
    end = min(parse_time(job.end), dur) if job.end else dur
    return max(0.0, end - start)


def plan_tasks(jobs, workers, split=True, min_chunk_sec=60.0, no_split=False):
    """jobs: list of Job. Returns [(job_index, Job-with-chunk)].
    When there are fewer jobs than workers, long jobs are split into chunks
    (never shorter than min_chunk_sec) so every worker has something to do."""
    tasks = [(i, j) for i, j in enumerate(jobs)]
    if not split or workers <= 1 or not jobs:
        return tasks
    durs = [_quick_duration(j) for j in jobs]
    total = sum(durs)
    if total <= 0:
        return tasks
    target = max(workers * 2, len(jobs))     # ~2 tasks per worker keeps them balanced
    out = []
    for i, (j, d) in enumerate(zip(jobs, durs)):
        n = max(1, min(int(round(target * d / total)), int(d // min_chunk_sec)))
        if no_split or j.frames:
            n = 1   # ROI tracking needs one continuous pass; marker lists are short
        out.extend((i, replace(j, chunk=k, chunks=n)) for k in range(n))
    return out


def worker_threads(workers):
    return max(1, (os.cpu_count() or 2) // max(1, workers))


def merge_stats(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out[k] = out.get(k, v) if k != "cancelled" else (out.get(k) or v)
        elif k == "scenes":
            out[k] = out.get(k, 0) + v - (1 if k in a else 0)
        else:
            out[k] = out.get(k, 0) + v
    return out


def run_batch(jobs, settings, workers=2, split=True, min_chunk_sec=60.0, on_event=None, cancel=None,
              threads_per_worker=0):
    """Blocking runner. on_event(kind, job_index, payload) for 'progress'
    (done, total, saved), 'done' (merged stats), 'error' (message).
    Returns {job_index: stats | Exception-string}."""
    tasks = plan_tasks(jobs, workers, split and not settings.roi_track and "{counter" not in settings.name_template,
                       min_chunk_sec)
    settings = replace(settings, det_threads=settings.det_threads or worker_threads(workers))
    from AIO_QC.core import procs
    ctx = mp.get_context("spawn")
    results = {}
    threads = max(1, threads_per_worker or procs.cpu_budget() // max(1, workers))
    manager = procs.start_manager(ctx)
    ex = None
    cancel_event = None
    try:
        q = manager.Queue()
        cancel_event = manager.Event()
        prog = {t: (0, 0, 0) for t in range(len(tasks))}
        pending = {i: sum(1 for ji, _ in tasks if ji == i) for i in range(len(jobs))}
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=procs.worker_init,
                                 initargs=(threads, True))
        if True:
            futs = [ex.submit(job_process, t, job, settings, q, cancel_event) for t, (_, job) in enumerate(tasks)]
            finished = 0
            while finished < len(tasks):
                if cancel is not None and cancel():
                    cancel_event.set()
                try:
                    msg = q.get(timeout=0.2)
                except Exception:
                    if all(f.done() for f in futs) and q.empty():
                        break
                    continue
                kind, t = msg[0], msg[1]
                ji = tasks[t][0]
                if kind == "progress":
                    prog[t] = (msg[2], msg[3], msg[4])
                    parts = [prog[x] for x, (jj, _) in enumerate(tasks) if jj == ji]
                    if on_event:
                        on_event("progress", ji, tuple(sum(p[i] for p in parts) for i in range(3)))
                elif kind == "done":
                    finished += 1
                    r = results.get(ji)
                    if not isinstance(r, str):          # an earlier chunk error sticks
                        results[ji] = msg[2] if r is None else merge_stats(r, msg[2])
                    pending[ji] -= 1
                    if pending[ji] == 0 and on_event:
                        on_event("done", ji, results[ji])
                elif kind == "error":
                    finished += 1
                    pending[ji] -= 1
                    results[ji] = msg[2]
                    if on_event:
                        on_event("error", ji, msg[2])
    except BaseException:
        try:
            if cancel_event is not None:
                cancel_event.set()             # workers checkpoint and stop at the next frame
        except Exception:
            pass
        procs.terminate_pool(ex, grace=3.0)    # Ctrl+C / error: don't wait for whole jobs to finish
        ex = None
        raise
    finally:
        if ex is not None:
            ex.shutdown(wait=True)
        try:
            manager.shutdown()
        except Exception:
            pass
    return results
