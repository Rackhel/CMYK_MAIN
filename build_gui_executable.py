#!/usr/bin/env python
"""
Build script: Package Grayspot GUI as executable using PyInstaller.
PyInstaller로 Grayspot GUI 실행 파일을 패키징하는 빌드 스크립트.

Generates standalone .exe (Windows) / .app (macOS) / binary (Linux)
without requiring Python installation.
설치된 Python 없이 실행 가능한 독립 실행 파일을 생성합니다.

Usage:
    # Default: inference mode, gui_for_user entry
    # 기본값: inference 모드, gui_for_user 진입점
    python build_gui_executable.py

    # Include model checkpoints for inference
    # 모델 체크포인트 포함 (inference 모드)
    python build_gui_executable.py --mode inference

    # Code only, no data/models
    # 코드만 포함, 데이터/모델 제외
    python build_gui_executable.py --mode shell --entry gui

    # Include all code + data/models for development/testing
    # 전체 코드 + 데이터/모델 포함 (개발/테스트용)
    python build_gui_executable.py --mode full

    # Create a single-file executable (may increase startup time)
    # 단일 파일 실행 파일 생성 (시작 시간이 길어질 수 있음)
    python build_gui_executable.py --onefile

Output:
    dist/grayspot/grayspot.exe  (Windows)
    dist/grayspot/grayspot      (Linux)
    dist/Grayspot.app           (macOS)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller  # noqa: F401

        return True
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PyInstaller executable for the CMYK GUI."
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "shell", "full"],
        default="inference",
        help=(
            "Packaging mode: inference (default), shell (no weights/data), "
            "or full (all data + code). "
            "기본값은 inference이며, shell은 코드만, full은 전체 패키징입니다."
        ),
    )
    parser.add_argument(
        "--entry",
        choices=["gui", "gui_for_user"],
        default="gui_for_user",
        help=(
            "Which GUI entrypoint to package: gui or gui_for_user. "
            "패키징할 GUI 진입점을 선택합니다."
        ),
    )
    parser.add_argument(
        "--name",
        default="grayspot",
        help="Executable base name. 실행 파일 기본 이름입니다.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help=(
            "Build a single-file executable instead of a one-folder bundle. "
            "단일 파일 실행 파일을 생성합니다."
        ),
    )
    return parser.parse_args()


def add_data_arg(root: Path, source: Path, dest: str) -> str:
    # PyInstaller expects add-data in the format <src><pathsep><dest>.
    # Windows는 ;, macOS/Linux는 :를 자동으로 처리합니다.
    separator = os.pathsep
    return f"{source}{separator}{dest}"


def list_model_dirs(root: Path) -> list[tuple[Path, str]]:
    """Return available model directories to include in inference mode.

    inference 모드에서 포함할 수 있는 모델 디렉토리를 반환합니다.
    """
    model_dirs = []
    for rel in ["data_set/models", "data_set/baseline"]:
        directory = root / rel
        if directory.exists():
            model_dirs.append((directory, rel))
    return model_dirs


def build_executable(args: argparse.Namespace) -> bool:
    """Build the PyInstaller command and run it.

    Args:
        args: Parsed command-line arguments.

    Returns:
        True if PyInstaller succeeded, False otherwise.
    """
    root = Path(__file__).parent
    entry_script = root / (
        "gui_for_user/main.py" if args.entry == "gui_for_user" else "gui/main.py"
    )

    icon_path = root / "gui/assets/icon.ico"
    if not icon_path.exists():
        icon_path = None

    print("🔨 Building Grayspot GUI Executable...")
    print(f"   Root: {root}")
    print(f"   Mode: {args.mode}")
    print(f"   Entrypoint: {entry_script}")

    add_data = [
        add_data_arg(root, root / "src", "src"),
        add_data_arg(root, root / "gui", "gui"),
        add_data_arg(root, root / "gui_for_user", "gui_for_user"),
        add_data_arg(root, root / "gui/assets", "gui/assets"),
        add_data_arg(root, root / "gui_for_user/assets", "gui_for_user/assets"),
    ]

    if args.mode == "inference":
        # inference 모드는 모델 체크포인트를 함께 포함합니다.
        for model_dir, dest in list_model_dirs(root):
            add_data.append(add_data_arg(root, model_dir, dest))
    elif args.mode == "full":
        # full 모드는 전체 데이터셋과 데이터 파이프라인 자산을 포함합니다.
        add_data.append(add_data_arg(root, root / "data_set", "data_set"))
        add_data.append(add_data_arg(root, root / "data", "data"))

    cmd = [sys.executable, "-m", "PyInstaller"]
    if args.onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    cmd.extend(["--windowed", "--name", args.name])

    if icon_path is not None:
        cmd.extend(["--icon", str(icon_path)])

    for item in add_data:
        cmd.extend(["--add-data", item])

    cmd.append(str(entry_script))

    print(f"\n📦 Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Build successful!")
        print(f"   Output: {root / 'dist' / args.name}")
        print("\n💡 To run the executable:")
        print(f"   - Windows: dist/{args.name}/{args.name}.exe")
        print(f"   - Linux:   dist/{args.name}/{args.name}")
        print(f"   - macOS:   dist/{args.name}.app")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        return False


if __name__ == "__main__":
    if check_pyinstaller():
        args = parse_arguments()
        success = build_executable(args)
        sys.exit(0 if success else 1)
