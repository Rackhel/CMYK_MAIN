# CMYK PyQt6 Desktop GUI Quick Start

## Local Setup

```bash
pip install -r requirements.txt
python main.py
```

Alternative entry point:

```bash
python -m gui.main
```

## PyInstaller

```bash
pip install pyinstaller
python build_gui_executable.py
python build_gui_executable.py --mode inference
python build_gui_executable.py --mode shell --entry gui
python build_gui_executable.py --mode full
```

- `--mode inference`: builds an inference executable. (recommended)
- `--mode shell`: packages only code and GUI, excluding models/data.
- `--mode full`: packages all code, models, and dataset.
- `--entry gui_for_user`: uses the gui_for_user app as the entrypoint.
- `--entry gui`: uses the original gui entrypoint.

---

- `--mode inference`: inference 실행 파일을 생성합니다. (추천)
- `--mode shell`: 코드와 GUI만 패키징하고, 모델/데이터는 제외합니다.
- `--mode full`: 전체 코드, 모델, 데이터셋을 모두 패키징합니다.
- `--entry gui_for_user`: `gui_for_user` 앱을 엔트리로 사용합니다.
- `--entry gui`: 기존 `gui` 엔트리를 사용합니다.

## GUI Structure

```text
gui/
├── main.py
├── main_window.py
├── workers/
├── services/
├── tabs/
├── components/
├── dialogs/
├── styles/
├── resources/
└── utils/
```

## Architecture Notes

- PyQt6 desktop application only.
- `MainWindow` owns tab orchestration and high-level signal routing.
- Tabs collect input and display results.
- Services create worker boundaries.
- Workers run long operations on QThread and communicate only through Qt signals.
- Backend functionality in `src/` stays black-boxed from the GUI.
