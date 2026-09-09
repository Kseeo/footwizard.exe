"""발 추출 마법사의 Flask 서버 -- wizard.html이 호출하는 API 라우트 전부.

"전처리" 탭(업로드 -> 방향 선택 -> 절단 위치 -> 완료)과 "GNN 구동" 탭(체크포인트
선택 후 하중 변형 예측)으로 구성. GNN 추론은 gnn_predict.py 참고.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from flask import Flask, jsonify, render_template, request, send_from_directory

import gnn_predict

# ---------------------------------------------------------------------------
# 경로 설정 -- 필요하면 여기만 고치면 됨.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FOOT_ENGINE_SRC = REPO_ROOT / "src"
if str(FOOT_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(FOOT_ENGINE_SRC))

# 비-UTF8 콘솔(cp949 등)에서 한글 출력이 깨지지 않도록.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

# exe(frozen) 상태에서는 워커를 별도 python.exe로 못 띄워 exe 자신을 재귀
# 호출한다(launcher.py 참고). 쓰기 가능한 _APP_DIR(jobs/)과 읽기 전용 번들
# 자산 경로 _BUNDLE_DIR(templates/)도 exe에서는 서로 다른 위치라 구분한다.
FROZEN = getattr(sys, "frozen", False)
_APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
_BUNDLE_DIR = Path(sys._MEIPASS) if FROZEN else Path(__file__).resolve().parent  # type: ignore[attr-defined]

JOBS_DIR = _APP_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

from foot_engine.stl_foot_extract.finishing import smooth_boundary_loops  # noqa: E402
from foot_engine.stl_foot_extract.postprocess_pipeline import (  # noqa: E402
    align_for_manual_cut,
    cut_and_finish_mesh,
)
STAGE1_ORIENTATION_WORKER = Path(__file__).resolve().parent / "stage1_orientation_worker.py"


def _stage1_orientation_worker_cmd() -> list[str]:
    if FROZEN:
        return [sys.executable, "--stage1-orientation-worker"]
    return [sys.executable, str(STAGE1_ORIENTATION_WORKER)]

# exe에서는 Flask의 기본 template_folder 추정이 안 맞아 명시적으로 지정.
app = Flask(__name__, template_folder=str(_BUNDLE_DIR / "templates")) if FROZEN else Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB


def job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise FileNotFoundError(f"알 수 없는 job_id: {job_id}")
    return d


# ---------------------------------------------------------------------------
# 작업 이력 -- job 폴더마다 진행 상태를 meta.json에 남겨 서버 재시작 후에도 유지.
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_meta(d: Path) -> dict:
    p = d / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_meta(d: Path, **updates) -> dict:
    meta = read_meta(d)
    meta.update(updates)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------

@app.route("/")
def wizard():
    """사용자용 마법사 화면(업로드 -> 방향 선택 -> 절단 -> 저장), 한 번에 한 단계만 크게."""
    return render_template("wizard.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="파일이 없습니다"), 400

    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    suffix = Path(f.filename).suffix or ".glb"
    input_path = d / f"0_input{suffix}"
    f.save(input_path)
    write_meta(d, original_filename=f.filename, created_at=_now_iso(), stage="uploaded")
    return jsonify(job_id=job_id, filename=input_path.name)


@app.route("/api/stage1_orientation/<job_id>", methods=["POST"])
def api_stage1_orientation(job_id):
    """발을 크롭하고 발바닥 방향 후보들을 미리보기 이미지로 반환 -- 사람이 눈으로 고른다."""
    try:
        d = job_dir(job_id)
        inputs = list(d.glob("0_input.*"))
        if not inputs:
            return jsonify(error="0단계(업로드) 결과가 없습니다"), 400

        out_cropped = d / "1a_cropped.glb"
        result_json = d / "1a_orientation_result.json"
        cmd = _stage1_orientation_worker_cmd() + [
            "--input", str(inputs[0]), "--output_cropped", str(out_cropped),
            "--job_dir", str(d), "--result_json", str(result_json),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        log = f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0 or not result_json.exists():
            return jsonify(error="방향 후보 워커가 실패했습니다(렌더링 크래시 가능성)", log=log), 500

        info = json.loads(result_json.read_text(encoding="utf-8"))
        return jsonify(**info)
    except subprocess.TimeoutExpired:
        return jsonify(error="방향 후보 계산 시간 초과(10분)"), 500
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/align_for_cut/<job_id>", methods=["POST"])
def api_align_for_cut(job_id):
    """3단계: 고른 방향으로 정렬만 한다(자르기/스케일 전) -- 절단 슬라이더가 보여줄 대상."""
    try:
        d = job_dir(job_id)
        args = request.get_json(silent=True) or {}
        cropped_file = args.get("cropped_file")
        down_direction = args.get("down_direction")
        if not cropped_file or not down_direction:
            return jsonify(error="cropped_file과 down_direction이 필요합니다"), 400

        mesh = trimesh.load(d / cropped_file, force="mesh", process=False)
        aligned = align_for_manual_cut(mesh, down_direction=np.array(down_direction, dtype=np.float64))

        # 실제 절단은 이 고해상도 원본으로 한다.
        out_path = d / "3_aligned_for_cut.glb"
        aligned.export(out_path)

        # 3단계 뷰어용 미리보기는 텍스처 없이 가볍게(정점 감소 전에 절단면
        # 테두리부터 다듬어야 톱니 모양이 덜 도드라진다).
        preview_path = d / "3_preview.glb"
        preview_source = smooth_boundary_loops(aligned)
        preview_faces = min(len(preview_source.faces), 6000)
        if len(preview_source.faces) > preview_faces:
            preview = preview_source.simplify_quadric_decimation(face_count=preview_faces)
        else:
            preview = preview_source.copy()
        preview.visual = trimesh.visual.ColorVisuals(mesh=preview, vertex_colors=[200, 170, 150, 255])
        preview.export(preview_path)

        y = aligned.vertices[:, 1]
        return jsonify(
            file=out_path.name,
            preview_file=preview_path.name,
            n_vertices=len(aligned.vertices),
            y_min=float(y.min()),
            y_max=float(y.max()),
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/cut_and_save/<job_id>", methods=["POST"])
def api_cut_and_save(job_id):
    """4단계("저장"): 고른 Y 높이에서 잘라 스케일+정리+스무딩까지 한 번에 마친다."""
    try:
        d = job_dir(job_id)
        args = request.get_json(silent=True) or {}
        cut_y = args.get("y")
        if cut_y is None:
            return jsonify(error="y가 필요합니다"), 400
        reference_length_mm = args.get("reference_length_mm")
        quat = args.get("quat")  # three.js Quaternion.toArray() = [x,y,z,w]

        aligned = trimesh.load(d / "3_aligned_for_cut.glb", force="mesh", process=False)
        if quat:
            # three.js가 쓴 쿼터니언을 그대로 적용(각도 재계산 없이 뷰어와 저장
            # 결과가 어긋나지 않게) -- trimesh는 [w,x,y,z] 순서라 재배열.
            x, y, z, w = quat
            rot = trimesh.transformations.quaternion_matrix([w, x, y, z])
            aligned.apply_transform(rot)
        result = cut_and_finish_mesh(
            aligned, cut_y=float(cut_y), reference_length_mm=reference_length_mm,
            z_up=False, floor_contact_tolerance_mm=2.0,
        )

        out_path = d / "4_final.glb"
        result.mesh.export(out_path)
        if result.floor_contact_mask is not None:
            np.save(d / "4_final_floor_contact.npy", result.floor_contact_mask)

        # stage="done"이 되면 "작업 이력" 탭(GET /api/jobs)에 나타난다.
        write_meta(
            d, stage="done", final_file=out_path.name,
            n_vertices=len(result.mesh.vertices), n_faces=len(result.mesh.faces),
            scale_factor=result.scale_factor, finished_at=_now_iso(),
        )

        return jsonify(
            file=out_path.name,
            n_vertices=len(result.mesh.vertices),
            n_faces=len(result.mesh.faces),
            scale_factor=result.scale_factor,
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/gnn_checkpoints", methods=["GET"])
def api_gnn_checkpoints():
    """GNN 패널의 모델(체크포인트) 선택 드롭다운용 목록."""
    try:
        return jsonify(checkpoints=gnn_predict.list_checkpoints())
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/send_to_gnn/<job_id>", methods=["POST"])
def api_send_to_gnn(job_id):
    """job_id 폴더 안의 메쉬(source_file, 기본 "4_final.glb")를 GNN으로 예측한다."""
    try:
        d = job_dir(job_id)
        args = request.get_json(silent=True) or {}
        source_file = Path(args.get("source_file") or "4_final.glb").name  # .name: job 폴더 밖을 못 가리키게
        src_path = d / source_file
        if not src_path.exists():
            return jsonify(error=f"입력 메쉬가 없습니다: {source_file}"), 400

        checkpoint = args.get("checkpoint")
        if not checkpoint:
            return jsonify(error="checkpoint가 필요합니다"), 400
        smooth = bool(args.get("smooth", True))
        lamb = args.get("lamb")
        iterations = args.get("iterations")
        floor_percentile = args.get("floor_percentile")

        result = gnn_predict.predict(
            src_path, d, checkpoint=checkpoint, smooth=smooth,
            lamb=float(lamb) if lamb is not None else None,
            iterations=int(iterations) if iterations is not None else None,
            floor_percentile=float(floor_percentile) if floor_percentile is not None else None,
        )

        write_meta(
            d, gnn_done=True, gnn_checkpoint=result["checkpoint"],
            gnn_predicted_file=result["predicted_file"],
            gnn_predicted_n_vertices=result["predicted_n_vertices"],
            gnn_predicted_n_faces=result["predicted_n_faces"],
            gnn_smoothed_file=result["smoothed_file"],
            gnn_smoothed_n_vertices=result["smoothed_n_vertices"],
            gnn_smoothed_n_faces=result["smoothed_n_faces"],
            gnn_at=_now_iso(),
        )

        return jsonify(**result)
    except gnn_predict.GnnPredictError as e:
        return jsonify(error=str(e), log=e.log), 502
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    """완료된(stage="done") 전처리 작업 목록 -- "GNN 구동" 탭 드롭다운용. 최신순."""
    jobs = []
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = read_meta(d)
        if meta.get("stage") != "done":
            continue
        jobs.append({
            "job_id": d.name,
            "filename": meta.get("original_filename"),
            "created_at": meta.get("created_at"),
            "finished_at": meta.get("finished_at"),
            "final_file": meta.get("final_file"),
            "n_vertices": meta.get("n_vertices"),
            "n_faces": meta.get("n_faces"),
            "gnn_done": bool(meta.get("gnn_done")),
            "gnn_checkpoint": meta.get("gnn_checkpoint"),
            "gnn_predicted_file": meta.get("gnn_predicted_file"),
            "gnn_predicted_n_vertices": meta.get("gnn_predicted_n_vertices"),
            "gnn_predicted_n_faces": meta.get("gnn_predicted_n_faces"),
            "gnn_smoothed_file": meta.get("gnn_smoothed_file"),
            "gnn_smoothed_n_vertices": meta.get("gnn_smoothed_n_vertices"),
            "gnn_smoothed_n_faces": meta.get("gnn_smoothed_n_faces"),
        })
    jobs.sort(key=lambda j: j.get("finished_at") or "", reverse=True)
    return jsonify(jobs=jobs)


@app.route("/jobs/<job_id>/<path:filename>")
def serve_job_file(job_id, filename):
    # ?download=1이면 attachment로 강제해 저장 다이얼로그가 뜨게 한다.
    as_attachment = request.args.get("download") == "1"
    return send_from_directory(job_dir(job_id), filename, as_attachment=as_attachment)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
