from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np


def set_platform_from_argv(
    default: str = "auto", *, deterministic_gpu_default: bool | None = None
) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default=default)
    if deterministic_gpu_default is not None:
        parser.add_argument(
            "--deterministic-gpu",
            action=argparse.BooleanOptionalAction,
            default=deterministic_gpu_default,
        )
    args, _ = parser.parse_known_args()
    if args.platform != "auto":
        os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else args.platform
    if getattr(args, "deterministic_gpu", False):
        flag = "--xla_gpu_deterministic_ops=true"
        flags = os.environ.get("XLA_FLAGS", "").split()
        if flag not in flags:
            os.environ["XLA_FLAGS"] = " ".join([*flags, flag])
    return args.platform


def configure_jax(platform: str, dtype: str) -> list[str]:
    import jax

    jax.config.update("jax_enable_x64", dtype == "float64")
    devices = jax.devices()
    if platform != "auto" and any(device.platform != platform for device in devices):
        raise RuntimeError(f"requested JAX platform {platform!r}, got {devices}")
    return [str(device) for device in devices]


def solver_source_fingerprint() -> tuple[str, list[str]]:
    package = Path(__file__).parents[1] / "src" / "jax_lbfgsb"
    paths = sorted(package.glob("*.py"))
    digest = hashlib.sha256()
    relative = []
    for path in paths:
        name = path.relative_to(package.parent.parent).as_posix()
        relative.append(name)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median_time(fn, repeats: int):
    import time

    values = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - started)
    return float(np.median(values)), values, result


def run_jax_chunks(solver, starts, batch_size: int, *payload):
    import jax
    import jax.numpy as jnp

    parts = []
    for offset in range(0, len(starts), batch_size):
        chunk = starts[offset : offset + batch_size]
        valid = len(chunk)
        if valid < batch_size:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], batch_size - valid, axis=0)]
            )
        result = solver.run(jnp.asarray(chunk), *payload)
        parts.append(jax.tree.map(lambda x: x[:valid], result))
    return jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)
