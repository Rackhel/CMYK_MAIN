"""Optuna HPO tab — HPO 실행 및 탐색 공간 전체 편집.
Contract: Contract_gui.md §3.5  /  SSOT_GUI.md §6.5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.components.log_panel import LogPanel
from gui.components.metric_card import MetricCard
from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.i18n import t
from gui.services.tuning_service import TuningService
from gui.tabs.base_tab import BaseTab
from gui.workers.tuning_worker import TuningWorker

_ROOT = Path(__file__).resolve().parents[2]
_SRC_CONFIG = _ROOT / "src" / "config" / "config.json"

# ── 레이아웃 상수 / Layout constants ─────────────────────────────────────────
_LABEL_W = 160
_FIELD_W = 180
_BS_W = 220  # batch_size / hidden_dim 쉼표 목록 필드


class OptunaTab(BaseTab):
    """Hyperparameter tuning controls, best-trial summary, and search-space editor."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = TuningService()
        self.worker: TuningWorker | None = None

        # ── 상단 컨트롤 / Top controls ─────────────────────────────────────
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K", "all"])
        self.channel_box.setMaximumWidth(80)

        self.phase_box = QComboBox()
        self.phase_box.addItems(["2 — Supervised", "0 — SimCLR"])
        self.phase_box.setMaximumWidth(130)

        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 500)
        self.trials_spin.setValue(self.cfg.get("optuna", {}).get("n_trials", 10))
        self.trials_spin.setMaximumWidth(80)

        self._start_btn = QPushButton(t("btn_start_hpo"))
        self._stop_btn = QPushButton(t("btn_stop"))
        self._start_btn.clicked.connect(self.start_tuning)
        self._stop_btn.clicked.connect(self.stop_tuning)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel"))
        ctrl_row.addWidget(self.channel_box)
        ctrl_row.addWidget(QLabel("Phase"))
        ctrl_row.addWidget(self.phase_box)
        ctrl_row.addWidget(QLabel("Trials"))
        ctrl_row.addWidget(self.trials_spin)
        ctrl_row.addWidget(self._start_btn)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch()

        # ── 결과 메트릭 카드 행 / Result metric cards ─────────────────────
        self.best_card = MetricCard("Best Value", "—")
        self.f1_card = MetricCard("Macro F1", "—")
        self.mae_card = MetricCard("MAE", "—")
        self.vacc_card = MetricCard("Val Acc", "—")
        self.tacc_card = MetricCard("Test Acc", "—")

        cards_row = QHBoxLayout()
        for c in (
            self.best_card,
            self.f1_card,
            self.mae_card,
            self.vacc_card,
            self.tacc_card,
        ):
            cards_row.addWidget(c)
        cards_row.addStretch()

        # ── Before / After 비교 패널 / Before-After comparison ────────────
        self._before_snapshot: dict[str, Any] = {}
        self._before_lbl = QLabel("(HPO 실행 전 스냅샷 없음)")
        self._after_lbl = QLabel("(결과 대기 중)")
        self._before_lbl.setWordWrap(True)
        self._after_lbl.setWordWrap(True)

        self._snapshot_btn = QPushButton(t("btn_snapshot"))
        self._snapshot_btn.clicked.connect(self._take_snapshot)

        self._compare_box = QGroupBox(t("grp_before_after"))
        c_h = QHBoxLayout(self._compare_box)
        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.addWidget(QLabel("<b>수정 전 (Before)</b>"))
        left_v.addWidget(self._before_lbl)
        left_v.addWidget(self._snapshot_btn)
        right_w = QWidget()
        right_v = QVBoxLayout(right_w)
        right_v.addWidget(QLabel("<b>수정 후 / HPO 결과 (After)</b>"))
        right_v.addWidget(self._after_lbl)
        c_h.addWidget(left_w)
        c_h.addWidget(right_w)

        # ── 비교 차트 / Comparison chart ──────────────────────────────────
        self._compare_chart = PlotlyWidget()
        self._compare_chart.setMinimumHeight(200)

        self.progress = ProgressPanel()

        # ── 탐색 공간 편집기 스크롤 / Search-space editor (scrollable) ────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        editor_layout = QVBoxLayout(scroll_content)
        editor_layout.setSpacing(10)
        editor_layout.setContentsMargins(4, 4, 4, 4)
        editor_layout.addWidget(self._build_global_group())
        editor_layout.addWidget(self._build_phase0_ss_group())
        editor_layout.addWidget(self._build_phase2_eff_group())
        editor_layout.addWidget(self._build_phase2_res_group())
        editor_layout.addStretch()
        scroll.setWidget(scroll_content)
        scroll.setMinimumHeight(300)

        # ── 저장 버튼 / Save / Reset ──────────────────────────────────────
        self.log_panel = LogPanel()
        self.log_panel.setMaximumHeight(80)
        self._save_btn = QPushButton(t("btn_save_ss"))
        self._reset_btn = QPushButton(t("btn_reset"))
        self._save_btn.clicked.connect(self.save_search_space)
        self._reset_btn.clicked.connect(self.refresh)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._reset_btn)

        # ── 탐색공간 편집기 패널 (스크롤 + 버튼 + 로그) / Editor panel ───
        editor_panel = QWidget()
        editor_v = QVBoxLayout(editor_panel)
        editor_v.setContentsMargins(0, 0, 0, 0)
        editor_v.setSpacing(4)
        self._editor_label = QLabel(t("grp_ss_editor"))
        editor_v.addWidget(self._editor_label)
        editor_v.addWidget(scroll, stretch=1)
        editor_v.addLayout(btn_row)
        editor_v.addWidget(self.log_panel)

        # ── HPO 실행 패널 / HPO run panel ─────────────────────────────────
        run_panel = QWidget()
        run_v = QVBoxLayout(run_panel)
        run_v.setContentsMargins(0, 0, 0, 0)
        run_v.setSpacing(6)
        run_v.addLayout(ctrl_row)
        run_v.addLayout(cards_row)
        run_v.addWidget(self._compare_box)
        run_v.addWidget(self._compare_chart)
        run_v.addWidget(self.progress, stretch=1)

        # ── QSplitter: 위=HPO실행, 아래=탐색공간편집기 ────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(run_panel)
        splitter.addWidget(editor_panel)
        splitter.setSizes([400, 600])
        splitter.setChildrenCollapsible(False)

        # ── 최상위 레이아웃 / Top-level layout ───────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        opt = self.cfg.get("optuna", {})
        self._n_trials.setValue(int(opt.get("n_trials", 10)))
        self._n_jobs.setValue(int(opt.get("n_jobs", 1)))
        self._sampler.setCurrentText(opt.get("sampler", "tpe"))
        pruner = opt.get("pruner", {})
        self._pruner_type.setCurrentText(pruner.get("type", "median"))
        self._pruner_warmup.setValue(int(pruner.get("n_warmup_steps", 10)))

        p0_ss = opt.get("phase0", {}).get("search_space", {})
        self._p0_lr_min.setValue(float(p0_ss.get("learning_rate", [1e-4, 1e-2])[0]))
        self._p0_lr_max.setValue(float(p0_ss.get("learning_rate", [1e-4, 1e-2])[1]))
        self._p0_wd_min.setValue(float(p0_ss.get("weight_decay", [1e-6, 1e-4])[0]))
        self._p0_wd_max.setValue(float(p0_ss.get("weight_decay", [1e-6, 1e-4])[1]))
        self._p0_ep_min.setValue(int(p0_ss.get("epochs", [5, 15])[0]))
        self._p0_ep_max.setValue(int(p0_ss.get("epochs", [5, 15])[1]))
        self._p0_bs.setText(self._csv_text(p0_ss.get("batch_size", [16, 32, 64])))
        p0_temp = p0_ss.get("temperature", [0.05, 0.5])
        self._p0_temp_min.setValue(float(p0_temp[0]))
        self._p0_temp_max.setValue(float(p0_temp[1]))
        p0_wp = p0_ss.get("warmup_epochs", [0, 5])
        self._p0_wp_min.setValue(int(p0_wp[0]))
        self._p0_wp_max.setValue(int(p0_wp[1]))
        self._p0_hd.setText(self._csv_text(p0_ss.get("hidden_dim") or [128, 256, 512]))
        self._p0_pd.setText(
            self._csv_text(p0_ss.get("projection_dim") or [64, 128, 256])
        )

        p2_ss = opt.get("phase2", {}).get("search_space", {})
        eff = p2_ss.get("efficientnet_b0", {})
        self._eff_lr_min.setValue(float(eff.get("learning_rate", [5e-5, 3e-4])[0]))
        self._eff_lr_max.setValue(float(eff.get("learning_rate", [5e-5, 3e-4])[1]))
        self._eff_wd_min.setValue(float(eff.get("weight_decay", [1e-4, 1e-3])[0]))
        self._eff_wd_max.setValue(float(eff.get("weight_decay", [1e-4, 1e-3])[1]))
        self._eff_ep_min.setValue(int(eff.get("epochs", [10, 30])[0]))
        self._eff_ep_max.setValue(int(eff.get("epochs", [10, 30])[1]))
        self._eff_do_min.setValue(float(eff.get("dropout", [0.1, 0.3])[0]))
        self._eff_do_max.setValue(float(eff.get("dropout", [0.1, 0.3])[1]))
        self._eff_bs.setText(self._csv_text(eff.get("batch_size", [16, 32, 64])))
        self._eff_hd.setText(self._csv_text(eff.get("hidden_dim") or [128, 256]))
        eff_ls = eff.get("label_smoothing", [0.0, 0.2])
        self._eff_ls_min.setValue(float(eff_ls[0]))
        self._eff_ls_max.setValue(float(eff_ls[1]))
        eff_wp = eff.get("warmup_epochs", [0, 5])
        self._eff_wp_min.setValue(int(eff_wp[0]))
        self._eff_wp_max.setValue(int(eff_wp[1]))
        self._eff_cw.setText(
            self._csv_text(eff.get("class_weights", ["none", "balanced"]))
        )
        self._eff_fb.setText(self._csv_text(eff.get("frozen_backbone", [False, True])))

        res = p2_ss.get("resnet50", {})
        self._res_lr_min.setValue(float(res.get("learning_rate", [1e-4, 5e-4])[0]))
        self._res_lr_max.setValue(float(res.get("learning_rate", [1e-4, 5e-4])[1]))
        self._res_wd_min.setValue(float(res.get("weight_decay", [1e-3, 1e-2])[0]))
        self._res_wd_max.setValue(float(res.get("weight_decay", [1e-3, 1e-2])[1]))
        self._res_ep_min.setValue(int(res.get("epochs", [10, 30])[0]))
        self._res_ep_max.setValue(int(res.get("epochs", [10, 30])[1]))
        self._res_do_min.setValue(float(res.get("dropout", [0.3, 0.5])[0]))
        self._res_do_max.setValue(float(res.get("dropout", [0.3, 0.5])[1]))
        self._res_bs.setText(self._csv_text(res.get("batch_size", [16, 32, 64])))
        self._res_hd.setText(self._csv_text(res.get("hidden_dim") or [256, 512]))
        self._res_md.setText(self._csv_text(res.get("mid_dim", [256, 512, 1024])))
        res_ls = res.get("label_smoothing", [0.0, 0.2])
        self._res_ls_min.setValue(float(res_ls[0]))
        self._res_ls_max.setValue(float(res_ls[1]))
        res_wp = res.get("warmup_epochs", [0, 5])
        self._res_wp_min.setValue(int(res_wp[0]))
        self._res_wp_max.setValue(int(res_wp[1]))
        self._res_cw.setText(
            self._csv_text(res.get("class_weights", ["none", "balanced"]))
        )
        self._res_fb.setText(self._csv_text(res.get("frozen_backbone", [False, True])))

        self.trials_spin.setValue(int(opt.get("n_trials", 10)))

    @staticmethod
    def _csv_text(items: Any) -> str:
        if items is None:
            return ""
        if isinstance(items, str):
            return items
        try:
            return ",".join(str(x) for x in items)
        except Exception:
            return str(items)

    def retranslate_ui(self, lang: str) -> None:
        self._start_btn.setText(t("btn_start_hpo"))
        self._stop_btn.setText(t("btn_stop"))
        self._save_btn.setText(t("btn_save_ss"))
        self._reset_btn.setText(t("btn_reset"))
        self._snapshot_btn.setText(t("btn_snapshot"))
        self._compare_box.setTitle(t("grp_before_after"))
        self._editor_label.setText(t("grp_ss_editor"))

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        bv = result.get("best_value", 0)
        bp = result.get("best_params", {})
        self.best_card.set_value(f"{bv:.4f}")
        self.f1_card.set_value(f"{result.get('macro_f1', 0):.3f}")
        self.mae_card.set_value(f"{result.get('mae', 0):.3f}")
        self.vacc_card.set_value(f"{result.get('val_acc', 0):.3f}")
        self.tacc_card.set_value(f"{result.get('test_acc', 0):.3f}")
        self.progress.append_log(f"Best params: {bp}")
        self._after_lbl.setText(
            f"Best Value: {bv:.4f}\n"
            f"Val Acc:    {result.get('val_acc', 0):.3f}\n"
            f"Test Acc:   {result.get('test_acc', 0):.3f}\n"
            f"Macro F1:   {result.get('macro_f1', 0):.3f}\n"
            f"MAE:        {result.get('mae', 0):.3f}\n"
            f"Params:     {bp}"
        )
        self._render_compare_chart(result)

    # ── Public API ────────────────────────────────────────────────────────────

    def _take_snapshot(self) -> None:
        """현재 config를 'Before' 스냅샷으로 저장."""
        p2 = self.cfg.get("phase2", {})
        p0 = self.cfg.get("phase0", {})
        self._before_snapshot = {
            "p2_lr": p2.get("learning_rate", 0),
            "p2_wd": p2.get("weight_decay", 0),
            "p2_ep": p2.get("epochs", 0),
            "p0_lr": p0.get("learning_rate", 0),
            "p0_ep": p0.get("epochs", 0),
        }
        self._before_lbl.setText(
            f"Phase2 LR:    {self._before_snapshot['p2_lr']}\n"
            f"Phase2 WD:    {self._before_snapshot['p2_wd']}\n"
            f"Phase2 Epoch: {self._before_snapshot['p2_ep']}\n"
            f"Phase0 LR:    {self._before_snapshot['p0_lr']}\n"
            f"Phase0 Epoch: {self._before_snapshot['p0_ep']}"
        )

    def _render_compare_chart(self, after: dict) -> None:
        if not self._before_snapshot:
            return
        try:
            import plotly.graph_objects as go

            keys = ["p2_lr", "p2_wd", "p2_ep"]
            labels = ["Phase2 LR", "Phase2 WD", "Phase2 Epochs"]
            before_vals = [float(self._before_snapshot.get(k, 0)) for k in keys]
            after_vals = [
                float(after.get("best_params", {}).get(k.replace("p2_", ""), 0))
                for k in keys
            ]

            fig = go.Figure(
                data=[
                    go.Bar(
                        name="Before", x=labels, y=before_vals, marker_color="#94a3b8"
                    ),
                    go.Bar(
                        name="After", x=labels, y=after_vals, marker_color="#60a5fa"
                    ),
                ]
            )
            fig.update_layout(
                barmode="group",
                title="파라미터 Before / After",
                template="plotly_dark",
                height=200,
                margin={"t": 40, "b": 40},
            )
            self._compare_chart.set_figure(fig)
        except Exception:
            pass

    def start_tuning(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.progress.append_log("⚠️ HPO already running — stop it first")
            return
        channel = self.channel_box.currentText()
        n_trials = self.trials_spin.value()
        phase = int(self.phase_box.currentText()[0])

        self.worker = self.service.start_tuning(self.cfg, channel, n_trials, phase)
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(
            lambda msg: self.progress.append_log(f"ERROR: {msg}")
        )
        self.worker.start()

    def stop_tuning(self) -> None:
        self.service.stop_tuning()
        self.progress.append_log("Tuning stopped.")

    def save_search_space(self) -> None:
        try:
            src_cfg: dict = json.loads(_SRC_CONFIG.read_text(encoding="utf-8"))
            opt = src_cfg.setdefault("optuna", {})

            opt["n_trials"] = self._n_trials.value()
            opt["n_jobs"] = self._n_jobs.value()
            opt["sampler"] = self._sampler.currentText()
            opt.setdefault("pruner", {}).update(
                {
                    "type": self._pruner_type.currentText(),
                    "n_warmup_steps": self._pruner_warmup.value(),
                }
            )
            opt.setdefault("phase0", {}).setdefault("search_space", {}).update(
                {
                    "learning_rate": [self._p0_lr_min.value(), self._p0_lr_max.value()],
                    "weight_decay": [self._p0_wd_min.value(), self._p0_wd_max.value()],
                    "epochs": [self._p0_ep_min.value(), self._p0_ep_max.value()],
                    "batch_size": self._parse_int_list(
                        self._p0_bs.text(), [16, 32, 64]
                    ),
                    "temperature": [
                        self._p0_temp_min.value(),
                        self._p0_temp_max.value(),
                    ],
                    "warmup_epochs": [self._p0_wp_min.value(), self._p0_wp_max.value()],
                    "hidden_dim": self._parse_int_list(
                        self._p0_hd.text(), [128, 256, 512]
                    ),
                    "projection_dim": self._parse_int_list(
                        self._p0_pd.text(), [64, 128, 256]
                    ),
                }
            )
            eff_ss = (
                opt.setdefault("phase2", {})
                .setdefault("search_space", {})
                .setdefault("efficientnet_b0", {})
            )
            eff_ss.update(
                {
                    "learning_rate": [
                        self._eff_lr_min.value(),
                        self._eff_lr_max.value(),
                    ],
                    "weight_decay": [
                        self._eff_wd_min.value(),
                        self._eff_wd_max.value(),
                    ],
                    "epochs": [self._eff_ep_min.value(), self._eff_ep_max.value()],
                    "dropout": [self._eff_do_min.value(), self._eff_do_max.value()],
                    "batch_size": self._parse_int_list(
                        self._eff_bs.text(), [16, 32, 64]
                    ),
                    "hidden_dim": self._parse_int_list(self._eff_hd.text(), [128, 256]),
                    "label_smoothing": [
                        self._eff_ls_min.value(),
                        self._eff_ls_max.value(),
                    ],
                    "warmup_epochs": [
                        self._eff_wp_min.value(),
                        self._eff_wp_max.value(),
                    ],
                    "class_weights": self._parse_str_list(
                        self._eff_cw.text(), ["none", "balanced"]
                    ),
                    "frozen_backbone": self._parse_bool_list(
                        self._eff_fb.text(), [False, True]
                    ),
                }
            )
            res_ss = opt["phase2"]["search_space"].setdefault("resnet50", {})
            res_ss.update(
                {
                    "learning_rate": [
                        self._res_lr_min.value(),
                        self._res_lr_max.value(),
                    ],
                    "weight_decay": [
                        self._res_wd_min.value(),
                        self._res_wd_max.value(),
                    ],
                    "epochs": [self._res_ep_min.value(), self._res_ep_max.value()],
                    "dropout": [self._res_do_min.value(), self._res_do_max.value()],
                    "batch_size": self._parse_int_list(
                        self._res_bs.text(), [16, 32, 64]
                    ),
                    "hidden_dim": self._parse_int_list(self._res_hd.text(), [256, 512]),
                    "mid_dim": self._parse_int_list(
                        self._res_md.text(), [256, 512, 1024]
                    ),
                    "label_smoothing": [
                        self._res_ls_min.value(),
                        self._res_ls_max.value(),
                    ],
                    "warmup_epochs": [
                        self._res_wp_min.value(),
                        self._res_wp_max.value(),
                    ],
                    "class_weights": self._parse_str_list(
                        self._res_cw.text(), ["none", "balanced"]
                    ),
                    "frozen_backbone": self._parse_bool_list(
                        self._res_fb.text(), [False, True]
                    ),
                }
            )

            _SRC_CONFIG.write_text(
                json.dumps(src_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.cfg.update(src_cfg)
            self.trials_spin.setValue(self._n_trials.value())
            self.log_panel.append("✅ Search space saved → src/config/config.json")
        except Exception as exc:
            self.log_panel.append(f"❌ Save failed: {exc}")

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_global_group(self) -> QGroupBox:
        g = QGroupBox(t("grp_optuna_global"))
        f = self._make_form(g)
        opt = self.cfg.get("optuna", {})
        pruner = opt.get("pruner", {})

        self._n_trials = self._spin(opt.get("n_trials", 10), 1, 500)
        self._n_jobs = self._spin(opt.get("n_jobs", 1), 1, 16)
        self._sampler = QComboBox()
        self._sampler.addItems(["tpe", "random", "cmaes"])
        self._sampler.setCurrentText(opt.get("sampler", "tpe"))
        self._sampler.setMaximumWidth(_FIELD_W)
        self._pruner_type = QComboBox()
        self._pruner_type.addItems(["median", "hyperband", "none"])
        self._pruner_type.setCurrentText(pruner.get("type", "median"))
        self._pruner_type.setMaximumWidth(_FIELD_W)
        self._pruner_warmup = self._spin(pruner.get("n_warmup_steps", 10), 0, 100)

        f.addRow("n_trials", self._n_trials)
        f.addRow("n_jobs", self._n_jobs)
        f.addRow("Sampler", self._sampler)
        f.addRow("Pruner Type", self._pruner_type)
        f.addRow("Pruner Warmup", self._pruner_warmup)
        return g

    def _build_phase0_ss_group(self) -> QGroupBox:
        g = QGroupBox(t("grp_phase0"))
        f = self._make_form(g)
        ss = self.cfg.get("optuna", {}).get("phase0", {}).get("search_space", {})

        lr = ss.get("learning_rate", [1e-4, 1e-2])
        wd = ss.get("weight_decay", [1e-6, 1e-4])
        ep = ss.get("epochs", [5, 15])

        self._p0_lr_min = self._dspin(lr[0], 1e-7, 1.0, 7)
        self._p0_lr_max = self._dspin(lr[1], 1e-7, 1.0, 7)
        self._p0_wd_min = self._dspin(wd[0], 1e-8, 1.0, 8)
        self._p0_wd_max = self._dspin(wd[1], 1e-8, 1.0, 8)
        self._p0_ep_min = self._spin(ep[0], 1, 200)
        self._p0_ep_max = self._spin(ep[1], 1, 200)
        self._p0_bs = self._bsedit(ss.get("batch_size", [16, 32, 64]))

        temp = ss.get("temperature", [0.05, 0.5])
        self._p0_temp_min = self._dspin(temp[0], 0.01, 1.0, 3)
        self._p0_temp_max = self._dspin(temp[1], 0.01, 1.0, 3)

        wp = ss.get("warmup_epochs", [0, 5])
        self._p0_wp_min = self._spin(wp[0], 0, 20)
        self._p0_wp_max = self._spin(wp[1], 0, 20)

        # ProjectionHead hidden_dim (Tier 3) — config에서 null일 수 있으므로 or fallback
        # ProjectionHead hidden_dim (Tier 3) — may be null in config, use or fallback
        self._p0_hd = self._bsedit(ss.get("hidden_dim") or [128, 256, 512])
        # ProjectionHead projection_dim (Tier 3)
        self._p0_pd = self._bsedit(ss.get("projection_dim") or [64, 128, 256])

        f.addRow("LR min", self._p0_lr_min)
        f.addRow("LR max", self._p0_lr_max)
        f.addRow("WD min", self._p0_wd_min)
        f.addRow("WD max", self._p0_wd_max)
        f.addRow("Epochs min", self._p0_ep_min)
        f.addRow("Epochs max", self._p0_ep_max)
        f.addRow("Batch sizes (csv)", self._p0_bs)
        f.addRow("Temp min", self._p0_temp_min)
        f.addRow("Temp max", self._p0_temp_max)
        f.addRow("Warmup min", self._p0_wp_min)
        f.addRow("Warmup max", self._p0_wp_max)
        f.addRow("Hidden Dims (csv)", self._p0_hd)
        f.addRow("Proj Dims (csv)", self._p0_pd)
        return g

    def _build_phase2_eff_group(self) -> QGroupBox:
        g = QGroupBox(t("grp_phase2"))
        f = self._make_form(g)
        ss = (
            self.cfg.get("optuna", {})
            .get("phase2", {})
            .get("search_space", {})
            .get("efficientnet_b0", {})
        )

        lr = ss.get("learning_rate", [5e-5, 3e-4])
        wd = ss.get("weight_decay", [1e-4, 1e-3])
        ep = ss.get("epochs", [10, 30])
        do = ss.get("dropout", [0.1, 0.3])

        self._eff_lr_min = self._dspin(lr[0], 1e-7, 1.0, 7)
        self._eff_lr_max = self._dspin(lr[1], 1e-7, 1.0, 7)
        self._eff_wd_min = self._dspin(wd[0], 1e-8, 1.0, 8)
        self._eff_wd_max = self._dspin(wd[1], 1e-8, 1.0, 8)
        self._eff_ep_min = self._spin(ep[0], 1, 500)
        self._eff_ep_max = self._spin(ep[1], 1, 500)
        self._eff_do_min = self._dspin(do[0], 0.0, 0.9, 2)
        self._eff_do_max = self._dspin(do[1], 0.0, 0.9, 2)
        self._eff_bs = self._bsedit(ss.get("batch_size", [16, 32, 64]))
        self._eff_hd = self._bsedit(ss.get("hidden_dim") or [128, 256])

        ls = ss.get("label_smoothing", [0.0, 0.2])
        self._eff_ls_min = self._dspin(ls[0], 0.0, 0.5, 2)
        self._eff_ls_max = self._dspin(ls[1], 0.0, 0.5, 2)

        wp2 = ss.get("warmup_epochs", [0, 5])
        self._eff_wp_min = self._spin(wp2[0], 0, 20)
        self._eff_wp_max = self._spin(wp2[1], 0, 20)

        self._eff_cw = self._bsedit(ss.get("class_weights", ["none", "balanced"]))
        self._eff_fb = self._bsedit(ss.get("frozen_backbone", [False, True]))

        f.addRow("LR min", self._eff_lr_min)
        f.addRow("LR max", self._eff_lr_max)
        f.addRow("WD min", self._eff_wd_min)
        f.addRow("WD max", self._eff_wd_max)
        f.addRow("Epochs min", self._eff_ep_min)
        f.addRow("Epochs max", self._eff_ep_max)
        f.addRow("Dropout min", self._eff_do_min)
        f.addRow("Dropout max", self._eff_do_max)
        f.addRow("Batch sizes (csv)", self._eff_bs)
        f.addRow("Hidden dims (csv)", self._eff_hd)
        f.addRow("Label Smooth min", self._eff_ls_min)
        f.addRow("Label Smooth max", self._eff_ls_max)
        f.addRow("Warmup min", self._eff_wp_min)
        f.addRow("Warmup max", self._eff_wp_max)
        f.addRow("Class Weights (csv)", self._eff_cw)
        f.addRow("Frozen Backbone (csv)", self._eff_fb)
        return g

    def _build_phase2_res_group(self) -> QGroupBox:
        g = QGroupBox(t("grp_phase2"))
        f = self._make_form(g)
        ss = (
            self.cfg.get("optuna", {})
            .get("phase2", {})
            .get("search_space", {})
            .get("resnet50", {})
        )

        lr = ss.get("learning_rate", [1e-4, 5e-4])
        wd = ss.get("weight_decay", [1e-3, 1e-2])
        ep = ss.get("epochs", [10, 30])
        do = ss.get("dropout", [0.3, 0.5])

        self._res_lr_min = self._dspin(lr[0], 1e-7, 1.0, 7)
        self._res_lr_max = self._dspin(lr[1], 1e-7, 1.0, 7)
        self._res_wd_min = self._dspin(wd[0], 1e-8, 1.0, 8)
        self._res_wd_max = self._dspin(wd[1], 1e-8, 1.0, 8)
        self._res_ep_min = self._spin(ep[0], 1, 500)
        self._res_ep_max = self._spin(ep[1], 1, 500)
        self._res_do_min = self._dspin(do[0], 0.0, 0.9, 2)
        self._res_do_max = self._dspin(do[1], 0.0, 0.9, 2)
        self._res_bs = self._bsedit(ss.get("batch_size", [16, 32, 64]))
        self._res_hd = self._bsedit(ss.get("hidden_dim") or [256, 512])
        self._res_md = self._bsedit(ss.get("mid_dim", [256, 512, 1024]))

        ls_r = ss.get("label_smoothing", [0.0, 0.2])
        self._res_ls_min = self._dspin(ls_r[0], 0.0, 0.5, 2)
        self._res_ls_max = self._dspin(ls_r[1], 0.0, 0.5, 2)

        wp2r = ss.get("warmup_epochs", [0, 5])
        self._res_wp_min = self._spin(wp2r[0], 0, 20)
        self._res_wp_max = self._spin(wp2r[1], 0, 20)

        self._res_cw = self._bsedit(ss.get("class_weights", ["none", "balanced"]))
        self._res_fb = self._bsedit(ss.get("frozen_backbone", [False, True]))

        f.addRow("LR min", self._res_lr_min)
        f.addRow("LR max", self._res_lr_max)
        f.addRow("WD min", self._res_wd_min)
        f.addRow("WD max", self._res_wd_max)
        f.addRow("Epochs min", self._res_ep_min)
        f.addRow("Epochs max", self._res_ep_max)
        f.addRow("Dropout min", self._res_do_min)
        f.addRow("Dropout max", self._res_do_max)
        f.addRow("Batch sizes (csv)", self._res_bs)
        f.addRow("Hidden dims (csv)", self._res_hd)
        f.addRow("Mid dims (csv)", self._res_md)
        f.addRow("Label Smooth min", self._res_ls_min)
        f.addRow("Label Smooth max", self._res_ls_max)
        f.addRow("Warmup min", self._res_wp_min)
        f.addRow("Warmup max", self._res_wp_max)
        f.addRow("Class Weights (csv)", self._res_cw)
        f.addRow("Frozen Backbone (csv)", self._res_fb)
        return g

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_form(parent: QGroupBox) -> QFormLayout:
        f = QFormLayout(parent)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        f.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        f.setContentsMargins(8, 6, 8, 6)
        return f

    @staticmethod
    def _spin(val, mn, mx, step=1) -> QSpinBox:
        w = QSpinBox()
        w.setRange(mn, mx)
        w.setSingleStep(step)
        w.setValue(int(val))
        w.setMaximumWidth(_FIELD_W)
        return w

    @staticmethod
    def _dspin(val, mn, mx, decimals=4) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(mn, mx)
        w.setDecimals(decimals)
        w.setSingleStep(10 ** (-decimals))
        w.setValue(float(val))
        w.setMaximumWidth(_FIELD_W)
        return w

    @staticmethod
    def _bsedit(items) -> QLineEdit:
        if items is None:
            text = ""
        elif isinstance(items, str):
            text = items
        else:
            try:
                text = ",".join(str(x) for x in items)
            except TypeError:
                text = str(items)

        w = QLineEdit(text)
        w.setMaximumWidth(_BS_W)
        return w

    @staticmethod
    def _parse_str_list(text: str, default: list) -> list:
        try:
            result = [x.strip() for x in text.split(",") if x.strip()]
            return result if result else default
        except Exception:
            return default

    @staticmethod
    def _parse_bool_list(text: str, default: list) -> list:
        try:
            mapping = {"true": True, "false": False, "1": True, "0": False}
            result = [
                mapping.get(x.strip().lower(), x.strip())
                for x in text.split(",")
                if x.strip()
            ]
            return result if result else default
        except Exception:
            return default

    @staticmethod
    def _parse_int_list(text: str, default: list) -> list:
        try:
            return [int(x.strip()) for x in text.split(",") if x.strip()]
        except Exception:
            return default
