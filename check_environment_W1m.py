#!/usr/bin/env python3
"""
Check whether the current Python environment can run the W1m pipeline.

The script scans the repository Python files for imports, reports which
third-party packages are installed, and checks external command-line programs
used by the pipeline.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path


PACKAGE_HINTS = {
    "astropy": "astropy",
    "astroquery": "astroquery",
    "donuts": "donuts",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "scipy": "scipy",
    "sep": "sep",
}

KNOWN_COMMAND_PATHS = {
    "solve-field": [
        "/opt/homebrew/bin/solve-field",
        "/usr/local/astrometry.net/bin/solve-field",
        "/usr/bin/solve-field",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report installed and missing dependencies for the W1m pipeline."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root to scan. Defaults to the directory containing this file.",
    )
    parser.add_argument(
        "--all-imports",
        action="store_true",
        help="Also print standard-library and local imports found during the scan.",
    )
    return parser.parse_args()


def stdlib_modules() -> set[str]:
    modules = set(sys.builtin_module_names)
    if hasattr(sys, "stdlib_module_names"):
        modules.update(sys.stdlib_module_names)
    return modules


def local_modules(root: Path) -> set[str]:
    modules = set()
    for py_file in root.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        modules.add(py_file.stem)
    for package_init in root.glob("*/__init__.py"):
        modules.add(package_init.parent.name)
    return modules


def imported_modules(py_file: Path) -> set[str]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    except SyntaxError as exc:
        print(f"WARNING: could not parse {py_file}: {exc}", file=sys.stderr)
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def scan_imports(root: Path) -> dict[str, set[str]]:
    results: dict[str, set[str]] = {}
    for py_file in sorted(root.glob("*.py")):
        if py_file.name == Path(__file__).name:
            continue
        modules = imported_modules(py_file)
        if modules:
            results[py_file.name] = modules
    return results


def import_version(module_name: str) -> str | None:
    package_name = PACKAGE_HINTS.get(module_name, module_name)
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        pass

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed, version unknown")


def check_python_packages(packages: set[str]) -> tuple[list[tuple[str, str | None]], list[str]]:
    installed = []
    missing = []

    for module_name in sorted(packages):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
            continue
        installed.append((module_name, import_version(module_name)))

    return installed, missing


def resolve_command(command: str) -> str | None:
    found = shutil.which(command)
    if found:
        return found

    for path in KNOWN_COMMAND_PATHS.get(command, []):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def command_version(path: str) -> str | None:
    for args in ([path, "--version"], [path, "-h"]):
        try:
            result = subprocess.run(
                args,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
        except Exception:
            continue
        output = result.stdout.strip().splitlines()
        if output:
            return output[0]
    return None


def print_list(title: str, rows: list[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  none")
        return
    for row in rows:
        print(f"  {row}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    root = args.root.resolve()
    imports_by_file = scan_imports(root)
    all_imports = set().union(*imports_by_file.values()) if imports_by_file else set()

    standard = stdlib_modules()
    local = local_modules(root)
    third_party = all_imports - standard - local

    print("W1m environment check")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Repository: {root}")

    installed, missing = check_python_packages(third_party)

    installed_rows = [
        f"{module}: {version or 'installed, version unknown'}" for module, version in installed
    ]
    missing_rows = [
        f"{module} (install hint: pip install {PACKAGE_HINTS.get(module, module)})"
        for module in missing
    ]
    print_list("Installed Python packages", installed_rows)
    print_list("Missing Python packages", missing_rows)

    command_rows = []
    missing_commands = []
    for command in sorted(KNOWN_COMMAND_PATHS):
        path = resolve_command(command)
        if path:
            version = command_version(path)
            command_rows.append(f"{command}: {path}" + (f" ({version})" if version else ""))
        else:
            missing_commands.append(
                f"{command} (required for astrometric solving; install Astrometry.net)"
            )

    print_list("Installed external commands", command_rows)
    print_list("Missing external commands", missing_commands)

    if args.all_imports:
        print_list("Standard-library imports", sorted(all_imports & standard))
        print_list("Local imports", sorted(all_imports & local))

    if missing or missing_commands:
        print("\nEnvironment is missing required dependencies.")
        if missing:
            packages = " ".join(PACKAGE_HINTS.get(module, module) for module in missing)
            print(f"Suggested Python install command: python -m pip install {packages}")
        return 1

    print("\nEnvironment looks ready for the W1m pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
