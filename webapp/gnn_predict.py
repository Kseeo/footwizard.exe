"""GNN 하중 변형 예측 -- 마법사 프로세스 안에서 hplAI 저장소의
build_dataset.py -> predict.py -> export_glb.py를 순서대로 subprocess 호출한다.

torch 환경(GNN_PYTHON)만 별도 conda라 그 경계는 여전히 subprocess다. 경로는
이 머신 기준 절대경로라 다른 머신이면 GNN_REPO_DIR/GNN_PYTHON/GNN_TRAIN_DATASET_*
환경변수로 맞춰야 한다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

GNN_REPO_DIR = Path(os.environ.get("GNN_REPO_DIR", "/home/hpl/ai/hplAI"))
GNN_PYTHON = os.environ.get("GNN_PYTHON", "/home/hpl/miniconda3/envs/mesh/bin/python")
GLB_PREPROCESS_DIR = GNN_REPO_DIR / "glb_preprocess"
CHECKPOINTS_DIR = GNN_REPO_DIR / "checkpoints_local"

# 학습 때 쓴 데이터셋 통계(정규화 등) -- 사용자가 고르는 값이 아니라 체크포인트에
# 고정된 조건. 다른 데이터셋으로 학습한 체크포인트를 쓰면 여기도 맞춰야 함.
DEFAULT_TRAIN_DATASET_PATH = os.environ.get(
    "GNN_TRAIN_DATASET_PATH", "/home/hpl/data/GNN_dataset/foot_all_1_frame"
)
DEFAULT_TRAIN_DATASET_FILE = os.environ.get("GNN_TRAIN_DATASET_FILE", "C3_bio.pt")
DEFAULT_TARGET_FACES = 6000

# hplAI의 foot_pipeline_postsmooth.py 기본값과 맞춘 것(라플라시안 스무딩 + 바닥 재접지).
SMOOTH_DEFAULTS = {"lamb": 0.5, "iterations": 10, "floor_percentile": 0.5}


class GnnPredictError(RuntimeError):
    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log


def list_checkpoints() -> list[dict]:
    """모델 선택 드롭다운용 -- checkpoints_local/*.pt 목록."""
    return [{"id": p.name, "label": p.stem} for p in sorted(CHECKPOINTS_DIR.glob("*.pt"))]


def _subprocess_env() -> dict:
    return {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "MGN_LOG_DIR": str(GNN_REPO_DIR / "checkpoints_local" / "mgn_logs") + os.sep,
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        "WANDB_MODE": "offline",
    }


def _run_step(cmd: list[str], log_lines: list[str]) -> None:
    log_lines.append("$ " + " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=str(GNN_REPO_DIR), capture_output=True, text=True, env=_subprocess_env(),
    )
    if result.stdout:
        log_lines.append(result.stdout)
    if result.stderr:
        log_lines.append(result.stderr)
    if result.returncode != 0:
        raise GnnPredictError(f"단계 실패(exit {result.returncode}): {' '.join(cmd)}", "\n".join(log_lines))


def _detect_up_axis(vertices: np.ndarray) -> int:
    """위쪽 축 = 최솟값이 0에 가장 가까운 축(메쉬가 바닥에 붙어 있다고 가정)."""
    mins = vertices.min(axis=0)
    return int(np.argmin(np.abs(mins)))


def _smooth_mesh(glb_path: Path, out_path: Path, lamb: float, iterations: int, floor_percentile: float) -> None:
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iterations)
    up_axis = _detect_up_axis(mesh.vertices)
    floor = np.percentile(mesh.vertices[:, up_axis], floor_percentile)
    mesh.vertices[:, up_axis] -= floor
    mesh.export(out_path)


def predict(
    input_glb: Path,
    out_dir: Path,
    checkpoint: str,
    target_faces: int = DEFAULT_TARGET_FACES,
    smooth: bool = True,
    lamb: float | None = None,
    iterations: int | None = None,
    floor_percentile: float | None = None,
) -> dict:
    """input_glb를 예측해 out_dir에 5_gnn_predicted.glb(및 smooth=True면 5_gnn_smoothed.glb)를 쓴다."""
    checkpoint_path = CHECKPOINTS_DIR / checkpoint
    if not checkpoint_path.is_file():
        raise GnnPredictError(f"체크포인트를 찾을 수 없습니다: {checkpoint}")

    lamb = SMOOTH_DEFAULTS["lamb"] if lamb is None else lamb
    iterations = SMOOTH_DEFAULTS["iterations"] if iterations is None else iterations
    floor_percentile = SMOOTH_DEFAULTS["floor_percentile"] if floor_percentile is None else floor_percentile

    work_dir = Path(tempfile.mkdtemp(prefix="wizard_gnn_"))
    log_lines: list[str] = []
    try:
        stem = "scan"
        input_dir = work_dir / "input"
        input_dir.mkdir()
        staged = input_dir / f"{stem}.glb"
        shutil.copyfile(input_glb, staged)

        scan_pt = work_dir / "scan.pt"
        pred_pt = work_dir / "predictions.pt"
        glb_out_dir = work_dir / "glb"

        _run_step([
            GNN_PYTHON, str(GLB_PREPROCESS_DIR / "build_dataset.py"),
            "--input_dir", str(input_dir), "--pattern", f"{stem}.glb",
            "--output", str(scan_pt), "--target_faces", str(target_faces),
        ], log_lines)

        _run_step([
            GNN_PYTHON, str(GLB_PREPROCESS_DIR / "predict.py"),
            "--checkpoint", str(checkpoint_path),
            "--train_dataset_path", DEFAULT_TRAIN_DATASET_PATH,
            "--train_dataset_file", DEFAULT_TRAIN_DATASET_FILE,
            "--scan_dataset", str(scan_pt), "--output", str(pred_pt),
        ], log_lines)

        _run_step([
            GNN_PYTHON, str(GLB_PREPROCESS_DIR / "export_glb.py"),
            "--predictions", str(pred_pt), "--output_dir", str(glb_out_dir),
        ], log_lines)

        predicted_src = glb_out_dir / f"{stem}_predicted.glb"
        if not predicted_src.is_file():
            raise GnnPredictError(f"예측 결과가 없습니다: {predicted_src}", "\n".join(log_lines))

        out_dir.mkdir(parents=True, exist_ok=True)
        predicted_name = "5_gnn_predicted.glb"
        shutil.copyfile(predicted_src, out_dir / predicted_name)
        predicted_mesh = trimesh.load(out_dir / predicted_name, force="mesh", process=False)

        result = {
            "checkpoint": checkpoint,
            "predicted_file": predicted_name,
            "predicted_n_vertices": len(predicted_mesh.vertices),
            "predicted_n_faces": len(predicted_mesh.faces),
            "smoothed_file": None,
            "smoothed_n_vertices": None,
            "smoothed_n_faces": None,
            "smooth_error": None,
            "log": "\n".join(log_lines)[-8000:],
        }

        if smooth:
            try:
                smoothed_name = "5_gnn_smoothed.glb"
                _smooth_mesh(out_dir / predicted_name, out_dir / smoothed_name, lamb, iterations, floor_percentile)
                smoothed_mesh = trimesh.load(out_dir / smoothed_name, force="mesh", process=False)
                result["smoothed_file"] = smoothed_name
                result["smoothed_n_vertices"] = len(smoothed_mesh.vertices)
                result["smoothed_n_faces"] = len(smoothed_mesh.faces)
            except Exception as e:  # noqa: BLE001
                # 스무딩 실패해도 예측 결과는 그대로 돌려준다.
                result["smooth_error"] = str(e)

        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
