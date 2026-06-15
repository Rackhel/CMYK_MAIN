"""
tuning/optuna_tuner.py

Optuna 기반 하이퍼파라미터 튜닝 모듈
Optuna-based hyperparameter tuning module

Phase 0 SimCLR 및 Phase 2 Supervised Classification 파이프라인을 재사용하여
각 Phase 하이퍼파라미터를 자동 탐색한다.
Reuses Phase 0 SimCLR and Phase 2 Supervised Classification pipelines
to automatically search hyperparameters for each phase.

지원 모드 / Supported modes:
    - Phase 0 단일/전체 채널 튜닝 / Phase 0 single/all-channel tuning
    - Phase 2 단일/전체 채널 튜닝 / Phase 2 single/all-channel tuning

Phase 0 탐색 대상 / Phase 0 search targets:
    - learning_rate, weight_decay, batch_size, epochs

Phase 2 탐색 대상 / Phase 2 search targets:
    - learning_rate, batch_size, weight_decay, epochs, dropout, hidden_dim, (mid_dim)

출력 / Outputs:
    outputs/optuna/
    ├── study_phase{N}_{channel}.db            ← Optuna 실험 DB / Optuna study database
    ├── best_params_phase{N}_{channel}.json    ← 최적 하이퍼파라미터 / Best hyperparameters
    └── trials_summary_phase{N}_{channel}.json ← 전체 trial 결과 요약 / All trial results summary

    최적 가중치 / Best weights (final retrain after optimization):
    Phase 0: data_set/models/phase0_backbone_{ch}_{tag}.pt
    Phase 2: data_set/models/best_{ch}.pt

실행 / Run:
    python -m src.scripts.run_optuna --phase 2
    python -m src.scripts.run_optuna --phase 0 --channel Y
    python -m src.scripts.run_optuna --phase 2 --channel all --trials 20
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import optuna
import torch

from src.tuning.search_space import get_phase0_search_space, get_phase2_search_space
from src.utils.optuna_utils import resolve_n_jobs, save_best_params, save_trials_summary

ROOT_DIR = Path(__file__).resolve().parents[2]


# Optuna SQLite storage는 thread-safe하나, 동시에 trial을 제출할 때
# 락 충돌을 줄이기 위해 trial 단위로 순차 제출을 보장하는 락을 사용.
# SQLite storage is thread-safe; this lock staggers concurrent trial submissions
# to reduce WAL contention under heavy parallel workloads.
_TRIAL_SUBMIT_LOCK = threading.Lock()


def _optimize_with_thread_pool(
    study: optuna.Study,
    objective_fn,
    n_trials: int,
    n_jobs: int,
) -> None:
    """
    ThreadPoolExecutor 기반 Optuna 병렬 최적화.
    Thread-pool-based Optuna parallel optimization.

    Optuna 기본 n_jobs 방식(joblib/fork)을 대체한다.
    Replaces Optuna's built-in n_jobs approach (joblib/fork).

    동작 방식 / How it works:
      - n_jobs=1 : 기존과 동일하게 study.optimize() 직접 호출 (오버헤드 없음)
                   Direct study.optimize() call — no overhead
      - n_jobs>1 : ThreadPoolExecutor(max_workers=n_jobs) 로 trial을 1개씩 제출.
                   각 스레드는 독립된 trial을 순차 실행하며 SQLite 공유 study에 기록.
                   Each thread runs one trial at a time against the shared SQLite study.

    장점 / Advantages over n_jobs:
      - Fork/spawn 없음 → macOS 메모리 압박 없음
        No fork/spawn → no macOS memory pressure
      - 스레드는 PyTorch MPS/CPU 텐서를 공유 메모리 없이 독립 실행 가능
        Threads run PyTorch MPS/CPU tensors independently without shared-memory issues
      - 풀 크기를 n_jobs로 명시적 제한 → 과도한 동시성 방지
        Pool size bounded by n_jobs → prevents runaway concurrency

    주의 / Note:
      - GIL로 인해 순수 Python 연산은 직렬화되지만, PyTorch 연산(C++ 커널)은
        GIL을 해제하므로 실질적 병렬 이득을 얻는다.
      - Due to GIL, pure Python is serialized, but PyTorch ops (C++ kernels)
        release the GIL and run in parallel.

    Args:
        study:        Optuna study 객체 / Optuna study object
        objective_fn: trial → float 목적 함수 / objective function
        n_trials:     총 trial 수 / total number of trials
        n_jobs:       동시 실행 스레드 수 / number of concurrent threads
    """
    if n_jobs <= 1:
        # 단일 스레드 경로 — 오버헤드 없음 / Single-thread path — zero overhead
        study.optimize(objective_fn, n_trials=n_trials, n_jobs=1)
        return

    def _run_one_trial(_: int) -> None:
        """1개의 trial을 실행하고 study에 기록. / Run one trial and record to study."""
        with _TRIAL_SUBMIT_LOCK:
            # 이미 n_trials 채워졌으면 제출 생략 / Skip if quota already met
            done = len(
                [
                    t
                    for t in study.trials
                    if t.state
                    in (
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                    )
                ]
            )
            if done >= n_trials:
                return
        # 락 밖에서 실제 학습 실행 (병렬성 확보)
        # Run actual training outside the lock (enables parallelism)
        study.optimize(objective_fn, n_trials=1, n_jobs=1)

    print(
        f"[ThreadPool] n_trials={n_trials}, n_jobs={n_jobs} "
        f"(fork-free thread pool / 프로세스 없는 스레드 풀)"
    )
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_run_one_trial, i) for i in range(n_trials)]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                print(f"[ThreadPool] Trial worker raised: {exc}")


# ──────────────────────────────────────────────────────────────
# Phase 0 Objective
# ──────────────────────────────────────────────────────────────


def objective_phase0(
    trial: optuna.Trial,
    channel: str,
) -> float:
    """
    Phase 0 Optuna 목적 함수 — InfoNCE loss 최소화
    Phase 0 Optuna objective — minimize InfoNCE loss
    """
    from src.scripts.run_phase0 import run_phase0
    from src.utils import load_config

    cfg = load_config()

    params = get_phase0_search_space(trial, cfg)

    cfg["phase0"]["learning_rate"] = params["learning_rate"]
    cfg["phase0"]["weight_decay"] = params["weight_decay"]
    cfg["phase0"]["batch_size"] = params["batch_size"]
    cfg["phase0"]["epochs"] = params["epochs"]

    device = torch.device(cfg["system"]["device"])

    if channel != "all":
        result = run_phase0(
            cfg, channel=channel.upper(), device=device, optuna_trial=trial
        )
        if result.get("skipped", False):
            raise optuna.exceptions.TrialPruned()
        return float(result["final_loss"])

    # All-channel: 채널별 final_loss 평균 반환 / return average final_loss across channels
    channels = ["Y", "M", "C", "K"]
    losses = []
    for ch in channels:
        result = run_phase0(cfg, channel=ch, device=device, optuna_trial=trial)
        if result.get("skipped", False):
            continue
        losses.append(result["final_loss"])

    if not losses:
        raise optuna.exceptions.TrialPruned()
    return float(sum(losses) / len(losses))


# ──────────────────────────────────────────────────────────────
# Phase 2 Objective
# ──────────────────────────────────────────────────────────────


def objective(
    trial: optuna.Trial,
    channel: str,
    phase0_dir: Path,
    ckpt_dir: Path,
) -> float:
    """
    Phase 2 Optuna 목적 함수 — val_acc 최대화
    Phase 2 Optuna objective — maximize validation accuracy
    """
    from src.scripts.run_phase2 import run_phase2
    from src.utils import load_config

    cfg = load_config()

    params = get_phase2_search_space(trial, cfg)

    cfg["phase2"]["learning_rate"] = params["learning_rate"]
    cfg["phase2"]["batch_size"] = params["batch_size"]
    cfg["phase2"]["weight_decay"] = params["weight_decay"]
    cfg["phase2"]["epochs"] = params["epochs"]

    backbone_name = cfg["model"]["backbone"]
    if "heads" not in cfg["phase2"]:
        cfg["phase2"]["heads"] = {}
    if backbone_name not in cfg["phase2"]["heads"]:
        cfg["phase2"]["heads"][backbone_name] = {}

    cfg["phase2"]["heads"][backbone_name]["dropout"] = params["dropout"]
    cfg["phase2"]["heads"][backbone_name]["hidden_dim"] = params["hidden_dim"]
    if "mid_dim" in params:
        cfg["phase2"]["heads"][backbone_name]["mid_dim"] = params["mid_dim"]

    device = torch.device(cfg["system"]["device"])

    if channel != "all":
        result = run_phase2(
            cfg,
            channel=channel.upper(),
            device=device,
            phase0_dir=phase0_dir,
            ckpt_dir=ckpt_dir,
            optuna_trial=trial,
        )
        if result.get("skipped", False):
            return 0.0
        return float(result["best_val_acc"])

    channels = ["Y", "M", "C", "K"]
    scores = []
    for ch in channels:
        result = run_phase2(
            cfg,
            channel=ch,
            device=device,
            phase0_dir=phase0_dir,
            ckpt_dir=ckpt_dir,
            optuna_trial=trial,
        )
        if result.get("skipped", False):
            continue
        scores.append(result["best_val_acc"])

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


# ──────────────────────────────────────────────────────────────
# Final retrain helpers
# ──────────────────────────────────────────────────────────────


def _retrain_phase0_best(best_params: dict, channel: str, cfg: dict) -> None:
    """
    Best Phase 0 params로 최종 재학습 후 backbone 저장.
    Final retrain with best Phase 0 params, then save backbone to models_dir.
    """
    from src.scripts.run_phase0 import run_phase0
    from src.utils.optuna_utils import apply_phase0_params

    final_cfg = apply_phase0_params(cfg, best_params)
    device = torch.device(final_cfg["system"]["device"])

    target_channels = ["Y", "M", "C", "K"] if channel == "all" else [channel.upper()]
    print(f"\n[Optuna] Phase 0 final retrain — channels: {target_channels}")

    for ch in target_channels:
        result = run_phase0(final_cfg, channel=ch, device=device)
        if result.get("skipped", False):
            print(f"  [{ch}] skipped (no data)")
        else:
            print(
                f"  [{ch}] final_loss={result['final_loss']:.4f}  backbone → {result['backbone_path']}"
            )


def _retrain_phase2_best(
    best_params: dict,
    channel: str,
    cfg: dict,
    phase0_dir: Path,
    models_dir: Path,
) -> None:
    """
    Best Phase 2 params로 최종 재학습 후 best_{ch}.pt를 models_dir에 저장.
    Final retrain with best Phase 2 params; saves best_{ch}.pt to models_dir.
    """
    from src.scripts.run_phase2 import run_phase2
    from src.utils.optuna_utils import apply_phase2_params

    final_cfg = apply_phase2_params(cfg, best_params)
    device = torch.device(final_cfg["system"]["device"])

    target_channels = ["Y", "M", "C", "K"] if channel == "all" else [channel.upper()]
    print(f"\n[Optuna] Phase 2 final retrain — channels: {target_channels}")

    for ch in target_channels:
        result = run_phase2(
            final_cfg,
            channel=ch,
            device=device,
            phase0_dir=phase0_dir,
            ckpt_dir=models_dir,
        )
        if result.get("skipped", False):
            print(f"  [{ch}] skipped (no data)")
        else:
            print(
                f"  [{ch}] test_acc={result['test_acc']:.4f}  "
                f"best_val_acc={result['best_val_acc']:.4f}  "
                f"→ {result['checkpoint_path']}"
            )


# ──────────────────────────────────────────────────────────────
# run_phase0_optuna
# ──────────────────────────────────────────────────────────────


def run_phase0_optuna(n_trials: int | None = None, channel: str = "all") -> None:
    """
    Phase 0 Optuna 하이퍼파라미터 튜닝 실행.
    Run Optuna hyperparameter optimization for Phase 0.

    최적화 완료 후 best params로 최종 재학습하여 backbone 가중치를 갱신한다.
    After optimization, retrains with best params to update backbone weights.
    """
    from src.utils import load_config

    channel = channel.lower()
    cfg = load_config()

    if not cfg.get("optuna", {}).get("enabled", True):
        print(
            "[Optuna] optuna.enabled=false — 튜닝이 비활성화되어 있습니다 / Tuning is disabled."
        )
        print("         config.json의 optuna.enabled를 true로 설정 후 실행하세요.")
        sys.exit(0)

    output_dir = ROOT_DIR / "outputs" / "optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_trials is None:
        n_trials = int(cfg.get("optuna", {}).get("n_trials", 5))

    seed = cfg["train"].get("seed", 42)
    sampler_name = cfg.get("optuna", {}).get("sampler", "tpe").lower()
    sampler = (
        optuna.samplers.RandomSampler(seed=seed)
        if sampler_name == "random"
        else optuna.samplers.TPESampler(seed=seed)
    )
    pruner_cfg = cfg.get("optuna", {}).get("pruner", {})
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 10))
    )

    # Phase 0 direction: minimize loss
    direction = cfg.get("optuna", {}).get("phase0", {}).get("direction", "minimize")

    study_suffix = f"phase0_{channel}"
    study_name = f"phase0_tuning_{channel}"
    storage_path = f"sqlite:///{output_dir}/study_{study_suffix}.db"

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective_fn = partial(objective_phase0, channel=channel)
    n_jobs = resolve_n_jobs(cfg)
    _optimize_with_thread_pool(study, objective_fn, n_trials=n_trials, n_jobs=n_jobs)

    print(f"\n[Phase 0 Optuna] Best Trial")
    print(f"  Channel  : {channel.upper()}")
    print(f"  Best Loss: {study.best_value:.4f}")
    print(f"  Params   : {study.best_trial.params}")

    save_best_params(study.best_trial.params, study_suffix, output_dir)
    save_trials_summary(study.trials, study_suffix, output_dir)

    # ── 최종 재학습 → backbone 가중치 갱신 / Final retrain → update backbone weights ──
    print("\n" + "=" * 60)
    print("  [Phase 0] Final retrain with best params")
    print("=" * 60)
    _retrain_phase0_best(
        best_params=study.best_trial.params,
        channel=channel,
        cfg=cfg,
    )


# ──────────────────────────────────────────────────────────────
# run_optuna (Phase 2)
# ──────────────────────────────────────────────────────────────


def run_optuna(n_trials: int | None = None, channel: str = "all") -> None:
    """
    Phase 2 Optuna 하이퍼파라미터 튜닝 실행.
    Run Optuna hyperparameter optimization for Phase 2.

    최적화 완료 후 best params로 최종 재학습하여 best_{ch}.pt를 갱신한다.
    After optimization, retrains with best params to update best_{ch}.pt.
    """
    from src.scripts.run_phase2 import (  # noqa: F401 (lazy — keep tuning→scripts boundary)
        run_phase2,
    )
    from src.utils import load_config

    channel = channel.lower()
    cfg = load_config()

    if not cfg.get("optuna", {}).get("enabled", True):
        print(
            "[Optuna] optuna.enabled=false — 튜닝이 비활성화되어 있습니다 / Tuning is disabled."
        )
        print("         config.json의 optuna.enabled를 true로 설정 후 실행하세요.")
        sys.exit(0)

    models_dir = ROOT_DIR / cfg["storage"]["models_dir"]
    phase0_dir = models_dir
    ckpt_dir = ROOT_DIR / "outputs" / "checkpoints"

    # Phase 0 backbone 존재 확인 / Check Phase 0 backbone existence
    target_channels = ["Y", "M", "C", "K"] if channel == "all" else [channel.upper()]
    try:
        from src.utils.utils_model import backbone_tag

        _tag = backbone_tag(cfg["model"]["backbone"])
    except Exception:
        _tag = cfg["model"]["backbone"].replace("_", "").replace("-", "")[:6]

    missing = [
        ch
        for ch in target_channels
        if not (phase0_dir / f"phase0_backbone_{ch}_{_tag}.pt").exists()
    ]
    if missing:
        print(
            f"[ERROR] Phase 0 backbone 없음 / not found: {missing}\n"
            f"        Run Phase 0 first: python -m src.scripts.run_phase0\n"
            f"        Path: {phase0_dir}"
        )
        sys.exit(1)

    output_dir = ROOT_DIR / "outputs" / "optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_trials is None:
        n_trials = int(cfg.get("optuna", {}).get("n_trials", 5))

    seed = cfg["train"].get("seed", 42)
    sampler_name = cfg.get("optuna", {}).get("sampler", "tpe").lower()
    sampler = (
        optuna.samplers.RandomSampler(seed=seed)
        if sampler_name == "random"
        else optuna.samplers.TPESampler(seed=seed)
    )
    pruner_cfg = cfg.get("optuna", {}).get("pruner", {})
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 10))
    )

    # Phase 2 direction: maximize val_acc
    direction = cfg.get("optuna", {}).get("phase2", {}).get("direction", "maximize")

    study_suffix = f"phase2_{channel}"
    study_name = f"phase2_tuning_{channel}"
    storage_path = f"sqlite:///{output_dir}/study_{study_suffix}.db"

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective_fn = partial(
        objective, channel=channel, phase0_dir=phase0_dir, ckpt_dir=ckpt_dir
    )
    n_jobs = resolve_n_jobs(cfg)
    _optimize_with_thread_pool(study, objective_fn, n_trials=n_trials, n_jobs=n_jobs)

    print(f"\n[Phase 2 Optuna] Best Trial")
    print(f"  Channel     : {channel.upper()}")
    print(f"  Best Val Acc: {study.best_value:.4f}")
    print(f"  Params      : {study.best_trial.params}")

    save_best_params(study.best_trial.params, study_suffix, output_dir)
    save_trials_summary(study.trials, study_suffix, output_dir)

    # ── 최종 재학습 → best_{ch}.pt 갱신 / Final retrain → update best_{ch}.pt ──
    print("\n" + "=" * 60)
    print("  [Phase 2] Final retrain with best params")
    print("=" * 60)
    _retrain_phase2_best(
        best_params=study.best_trial.params,
        channel=channel,
        cfg=cfg,
        phase0_dir=phase0_dir,
        models_dir=models_dir,
    )


if __name__ == "__main__":
    run_optuna()
