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


def locked_cuda_build(records: dict) -> str:
    """CUDA version of the one Linux torch build in uv.lock (2.13.0+cu126 -> 12.6)."""
    builds = {
        entry["version"].partition("+cu")[2]
        for entry in records.get("torch", [])
        if "+cu" in entry["version"]
    }
    if len(builds) != 1:
        raise SystemExit(
            "BLOCKED: uv.lock should lock one CUDA build of torch for Linux; "
            f"found {sorted(builds) or 'none'}."
        )
    digits = builds.pop()
    return f"{digits[:-1]}.{digits[-1]}"


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
        locked_cuda = locked_cuda_build(records)
        if torch.version.cuda != locked_cuda:
            raise SystemExit(
                f"BLOCKED: uv.lock selects torch built for CUDA {locked_cuda}, but "
                f"this environment has CUDA {torch.version.cuda}. Reconcile the "
                "environment through uv; do not silently replace torch with a module "
                "or pip install."
            )
        print(f"Locked Linux torch build: CUDA {locked_cuda}.")

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
    major, minor = torch.cuda.get_device_capability(0)
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        raise SystemExit(
            f"BLOCKED: this torch build has no kernels for sm_{major}{minor} "
            f"({torch.cuda.get_device_name(0)}); it supports "
            f"{torch.cuda.get_arch_list()}."
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
