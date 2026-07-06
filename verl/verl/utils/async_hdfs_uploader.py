# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process-local async HDFS uploader.

Training threads enqueue (src, dst) copy tasks and return immediately. A small
pool of background worker threads runs ``hdfs_io.makedirs`` + ``hdfs_io.copy``
with bounded retries, tracks per-path inflight state, and collects failures.

Designed as a single self-contained file so the integration can be removed for
open-sourcing by deleting this module plus the (small) call-site blocks in the
trainers / checkpoint manager. With the uploader disabled (``enable=False``),
``enqueue_copy`` falls back to a synchronous ``hdfs_io.copy`` so behaviour is
unchanged.

Public surface:

* ``init_uploader(config)``               - construct + start the singleton.
* ``get_uploader()``                      - return the singleton (no-op if uninit).
* ``shutdown_uploader(final_wait=True)``  - wait for all + stop workers.

* ``AsyncHdfsUploader.enqueue_copy(src, dst)``
* ``AsyncHdfsUploader.wait_for_path(path)``
* ``AsyncHdfsUploader.wait_for_all(timeout=None)``
* ``AsyncHdfsUploader.failures``  - list[FailedTask]

The config dict / OmegaConf node accepts:

    enable:        bool, default True (treated as False if no remote dst is ever enqueued)
    max_workers:   int,  default 2
    retry_count:   int,  default 3
    retry_backoff: float seconds between retries, default 5.0
    fail_on_error: bool, default False - if True, raise on shutdown when any task failed
"""

from __future__ import annotations

import logging
import os
import posixpath
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.utils import hdfs_io

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_HDFS_UPLOAD_LOG_LEVEL", "INFO"))


@dataclass
class UploadTask:
    task_id: int
    src: str
    dst: str
    attempts: int = 0


@dataclass
class FailedTask:
    task_id: int
    src: str
    dst: str
    error: str
    attempts: int


@dataclass
class _UploaderConfig:
    enable: bool = True
    max_workers: int = 2
    retry_count: int = 3
    retry_backoff: float = 5.0
    fail_on_error: bool = False


def _to_uploader_config(config: Any) -> _UploaderConfig:
    if config is None:
        return _UploaderConfig()

    def _get(key, default):
        if hasattr(config, "get"):
            try:
                return config.get(key, default)
            except TypeError:
                pass
        return getattr(config, key, default)

    return _UploaderConfig(
        enable=bool(_get("enable", True)),
        max_workers=max(1, int(_get("max_workers", 2))),
        retry_count=max(0, int(_get("retry_count", 3))),
        retry_backoff=float(_get("retry_backoff", 5.0)),
        fail_on_error=bool(_get("fail_on_error", False)),
    )


class AsyncHdfsUploader:
    """Bounded worker pool that runs ``hdfs_io.copy`` off the training thread."""

    _SENTINEL = object()

    def __init__(self, config: Optional[Any] = None):
        self._cfg = _to_uploader_config(config)
        self._queue: queue.Queue = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._inflight_by_src: dict[str, set[int]] = {}
        self._inflight_cv = threading.Condition(self._lock)
        self._failed: list[FailedTask] = []
        self._next_id = 0
        self._started = False
        self._stopped = False

    # ---------------- properties ----------------

    @property
    def enabled(self) -> bool:
        return self._cfg.enable

    @property
    def failures(self) -> list[FailedTask]:
        with self._lock:
            return list(self._failed)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._started or not self._cfg.enable:
            self._started = True
            return
        for i in range(self._cfg.max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"hdfs-uploader-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        self._started = True
        logger.info(
            "AsyncHdfsUploader started: max_workers=%d retry_count=%d fail_on_error=%s",
            self._cfg.max_workers,
            self._cfg.retry_count,
            self._cfg.fail_on_error,
        )

    def shutdown(self, final_wait: bool = True) -> list[FailedTask]:
        if self._stopped:
            return self.failures
        if final_wait:
            self.wait_for_all()
        for _ in self._workers:
            self._queue.put(self._SENTINEL)
        for t in self._workers:
            t.join(timeout=30.0)
        self._stopped = True
        if self._failed:
            logger.warning("AsyncHdfsUploader shutdown with %d failed tasks", len(self._failed))
            for f in self._failed:
                logger.warning("  failed task=%d src=%s dst=%s attempts=%d err=%s",
                               f.task_id, f.src, f.dst, f.attempts, f.error)
        return self.failures

    # ---------------- public API ----------------

    def enqueue_copy(self, src: str, dst: str) -> Optional[int]:
        """Enqueue a copy. Returns task id (or None if disabled / falls back to sync).

        If the uploader is disabled, ``hdfs_io.copy`` runs synchronously on the
        caller thread so behaviour matches the pre-uploader path.
        """
        if not src or not dst:
            return None
        if not self._cfg.enable:
            self._sync_copy(src, dst)
            return None
        if not self._started:
            self.start()
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            self._inflight_by_src.setdefault(src, set()).add(task_id)
        task = UploadTask(task_id=task_id, src=src, dst=dst)
        self._queue.put(task)
        logger.info("hdfs upload enqueued: id=%d src=%s dst=%s", task_id, src, dst)
        return task_id

    def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        """Block until queue is drained and inflight set is empty."""
        if not self._cfg.enable:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        # Drain queue: queue.join is not used because we manage inflight set ourselves.
        with self._inflight_cv:
            while self._queue.unfinished_tasks > 0 or any(self._inflight_by_src.values()):
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return False
                self._inflight_cv.wait(timeout=remaining)
        return True

    def wait_for_path(self, path: str, timeout: Optional[float] = None) -> bool:
        """Block until all enqueued uploads with ``src == path`` finish."""
        if not self._cfg.enable:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._inflight_cv:
            while self._inflight_by_src.get(path):
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return False
                self._inflight_cv.wait(timeout=remaining)
        return True

    # ---------------- worker plumbing ----------------

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._queue.task_done()
                return
            assert isinstance(item, UploadTask)
            try:
                self._run_task(item)
            except Exception as exc:  # last-resort guard so worker thread never dies
                logger.exception("hdfs upload worker crashed: %s", exc)
                self._mark_failed(item, repr(exc))
            finally:
                self._finalize(item)
                self._queue.task_done()

    def _run_task(self, task: UploadTask) -> None:
        last_err: Optional[BaseException] = None
        for attempt in range(self._cfg.retry_count + 1):
            task.attempts = attempt + 1
            try:
                self._do_copy(task.src, task.dst)
                logger.info(
                    "hdfs upload ok: id=%d attempts=%d src=%s dst=%s",
                    task.task_id, task.attempts, task.src, task.dst,
                )
                return
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "hdfs upload attempt %d/%d failed: id=%d src=%s dst=%s err=%s",
                    task.attempts, self._cfg.retry_count + 1,
                    task.task_id, task.src, task.dst, exc,
                )
                if attempt < self._cfg.retry_count:
                    time.sleep(self._cfg.retry_backoff)
        # exhausted retries
        self._mark_failed(task, repr(last_err) if last_err else "unknown")

    @staticmethod
    def _do_copy(src: str, dst: str) -> None:
        # For HDFS destinations we explicitly mkdir only the parent. If dst is
        # pre-created and src is a directory, `hdfs dfs -put src dst` nests the
        # source basename under dst (e.g. global_step_20/global_step_20).
        parent = posixpath.dirname(dst.rstrip("/"))
        if parent and parent != dst:
            try:
                hdfs_io.makedirs(parent, exist_ok=True)
            except Exception as e:
                # Non-fatal: HDFS makedirs may race with concurrent uploads.
                logger.debug("hdfs makedirs(%s) warning: %s", parent, e)
        ok = hdfs_io.copy(src=src, dst=dst)
        if ok is False:
            raise RuntimeError(f"hdfs_io.copy returned False for {src} -> {dst}")

    @staticmethod
    def _sync_copy(src: str, dst: str) -> None:
        parent = posixpath.dirname(dst.rstrip("/"))
        if parent and parent != dst:
            try:
                hdfs_io.makedirs(parent, exist_ok=True)
            except Exception:
                pass
        ok = hdfs_io.copy(src=src, dst=dst)
        if ok is False:
            raise RuntimeError(f"hdfs_io.copy returned False for {src} -> {dst}")

    def _mark_failed(self, task: UploadTask, err: str) -> None:
        with self._lock:
            self._failed.append(
                FailedTask(task_id=task.task_id, src=task.src, dst=task.dst, error=err, attempts=task.attempts)
            )

    def _finalize(self, task: UploadTask) -> None:
        with self._inflight_cv:
            srcs = self._inflight_by_src.get(task.src)
            if srcs is not None:
                srcs.discard(task.task_id)
                if not srcs:
                    self._inflight_by_src.pop(task.src, None)
            self._inflight_cv.notify_all()


# --------------------------- module singleton ---------------------------

_GLOBAL_UPLOADER: Optional[AsyncHdfsUploader] = None
_NOOP_UPLOADER: Optional[AsyncHdfsUploader] = None
_INIT_LOCK = threading.Lock()


def init_uploader(config: Any = None) -> AsyncHdfsUploader:
    """Construct (or return) the process-wide uploader.

    Safe to call multiple times; the first call wins. Returns the singleton.
    """
    global _GLOBAL_UPLOADER
    with _INIT_LOCK:
        if _GLOBAL_UPLOADER is not None:
            return _GLOBAL_UPLOADER
        uploader = AsyncHdfsUploader(config=config)
        uploader.start()
        _GLOBAL_UPLOADER = uploader
        return uploader


def get_uploader() -> AsyncHdfsUploader:
    """Return the global uploader (creates a disabled no-op uploader if never inited)."""
    global _GLOBAL_UPLOADER, _NOOP_UPLOADER
    if _GLOBAL_UPLOADER is not None:
        return _GLOBAL_UPLOADER
    with _INIT_LOCK:
        if _GLOBAL_UPLOADER is not None:
            return _GLOBAL_UPLOADER
        if _NOOP_UPLOADER is None:
            _NOOP_UPLOADER = AsyncHdfsUploader(config={"enable": False})
        return _NOOP_UPLOADER


def shutdown_uploader(final_wait: bool = True) -> list[FailedTask]:
    """Wait for inflight uploads and stop workers. Idempotent."""
    global _GLOBAL_UPLOADER
    with _INIT_LOCK:
        uploader = _GLOBAL_UPLOADER
        _GLOBAL_UPLOADER = None
    if uploader is None:
        return []
    return uploader.shutdown(final_wait=final_wait)


__all__ = [
    "AsyncHdfsUploader",
    "FailedTask",
    "UploadTask",
    "init_uploader",
    "get_uploader",
    "shutdown_uploader",
]
