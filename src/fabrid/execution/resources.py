from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from fabrid.artifacts.paths import ArtifactPaths
from fabrid.config import EnvironmentText, MemoryBytes


@dataclass
class ResourceMonitor:
    peak_rss_mb: MemoryBytes = 0
    peak_gpu_mb: MemoryBytes = 0

    def report(self, logger: logging.Logger, phase: EnvironmentText) -> None:
        process = psutil.Process()
        memory = psutil.virtual_memory()
        rss_mb = process.memory_info().rss / (1 << 20)
        self.peak_rss_mb = max(self.peak_rss_mb, int(rss_mb))
        logger.info(
            "resources phase=%s rss_mb=%.0f peak_rss_mb=%d ram_used_mb=%.0f "
            "ram_available_mb=%.0f cpu_percent=%.1f",
            phase,
            rss_mb,
            self.peak_rss_mb,
            (memory.total - memory.available) / (1 << 20),
            memory.available / (1 << 20),
            process.cpu_percent(interval=None),
        )
        if torch.cuda.is_available():
            gpu_name: EnvironmentText = torch.cuda.get_device_name(0)
            gpu_allocated_bytes: MemoryBytes = torch.cuda.memory_allocated(0)
            gpu_peak_bytes: MemoryBytes = torch.cuda.max_memory_allocated(0)
            self.peak_gpu_mb = max(self.peak_gpu_mb, int(gpu_peak_bytes / (1 << 20)))
            logger.info(
                "gpu phase=%s name=%s memory_used_mb=%.0f peak_gpu_mb=%d",
                phase,
                gpu_name,
                float(gpu_allocated_bytes) / (1 << 20),
                self.peak_gpu_mb,
            )


_MONITOR = ResourceMonitor()


def configure_campaign_logging(paths: ArtifactPaths) -> logging.Logger:
    log_path = paths.campaign_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path = log_path.resolve()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    root = logging.getLogger()
    if any(
        isinstance(existing, logging.FileHandler)
        and Path(existing.baseFilename).resolve() == resolved_log_path
        for existing in root.handlers
    ):
        return root
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return root


def report_resources(logger: logging.Logger, phase: EnvironmentText) -> None:
    _MONITOR.report(logger, phase)


def peak_rss_mb() -> MemoryBytes:
    return _MONITOR.peak_rss_mb


def peak_gpu_mb() -> MemoryBytes:
    return _MONITOR.peak_gpu_mb
