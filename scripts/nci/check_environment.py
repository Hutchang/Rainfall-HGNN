"""Check the prepared environment without installing packages or writing files."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock_bytes = (root / "uv.lock").read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    errors = []

    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        errors.append(
            "Use this checkout's .venv/bin/python, prepared with uv sync --locked."
        )
    requirement = project["project"]["requires-python"]
    if platform.python_version() not in SpecifierSet(requirement):
        errors.append(
            f"Python {platform.python_version()} does not satisfy {requirement}."
        )
    records = {}
    for package in lock["package"]:
        records.setdefault(canonicalize_name(package["name"]), []).append(package)

    for text in project["project"]["dependencies"]:
        dependency = Requirement(text)
        if dependency.marker and not dependency.marker.evaluate():
            continue
        name = canonicalize_name(dependency.name)
        locked_versions = {entry["version"] for entry in records.get(name, [])}
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing locked dependency: {name}.")
            continue
        if installed not in locked_versions or installed not in dependency.specifier:
            errors.append(
                f"{name}={installed} differs from pyproject/uv.lock "
                f"(locked: {sorted(locked_versions)})."
            )

    print(
        f"Python: {platform.python_version()} ({platform.system()} {platform.machine()})"
    )
    print(f"uv.lock SHA256: {hashlib.sha256(lock_bytes).hexdigest()}")
    if errors:
        raise SystemExit("BLOCKED:\n- " + "\n- ".join(errors))

    import torch

    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"Compiled CUDA architectures: {torch.cuda.get_arch_list()}")
    if platform.system() == "Linux":
        linux_cuda13 = any(
            dependency["name"].endswith("-cu13")
            for entry in records.get("torch", [])
            for dependency in entry.get("dependencies", [])
        )
        if linux_cuda13 and (
            torch.version.cuda is None or not torch.version.cuda.startswith("13.")
        ):
            raise SystemExit(
                "BLOCKED: uv.lock selects a CUDA 13 Linux build, but this environment "
                "does not provide it. Reconcile the environment through uv; do not "
                "silently replace torch with a module or pip install."
            )
        if linux_cuda13:
            print(
                "Gadi queue choice: dgxa100; gpuvolta/V100 is incompatible with this lock."
            )

    if not args.cuda:
        print(
            "CUDA runtime, project allocation and data-project access remain unchecked."
        )
        return
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("BLOCKED: CUDA probing requires a PBS GPU allocation.")
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise SystemExit(
            "BLOCKED: CUDA is unavailable in this Linux PBS job; inspect driver/build compatibility."
        )
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            "BLOCKED: this smoke runner expects exactly one visible allocated GPU."
        )
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; capability: {torch.cuda.get_device_capability(0)}"
    )
    if torch.version.cuda.startswith("13.") and torch.cuda.get_device_capability(0) < (
        7,
        5,
    ):
        raise SystemExit(
            "BLOCKED: CUDA 13 cannot run on Volta/V100; use the reviewed A100 queue."
        )
    # Device discovery can succeed even when kernels do not support the GPU/driver.
    try:
        sample = torch.ones((16, 16), device="cuda", requires_grad=True)
        (sample @ sample).sum().backward()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise SystemExit(f"BLOCKED: CUDA forward/backward probe failed: {exc}") from exc
    print("CUDA matrix multiplication and backward probe passed.")


if __name__ == "__main__":
    main()
