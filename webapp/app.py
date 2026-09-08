"""발 추출 마법사의 Flask 서버 -- 웹 UI(wizard.html)가 호출하는 API 라우트 전부.

`launcher.py`가 이 앱을 띄운다. 마법사는 화면 4단계로 진행된다:

  1단계(업로드): api_upload()
  2단계(방향 선택): api_stage1_orientation() -- 발 부위를 크롭하고
         (`crop_foot_mesh()`) 발바닥 방향 후보 몇 개를 계산해 사람이 고르게 한다.
  3단계(절단 위치): api_align_for_cut() -- 고른 방향으로 메쉬를 정렬한다.
         발목 절단 높이는 사람이 3D 뷰어를 보고 직접 고른다(자동 탐지 없음).
  4단계(완료, "저장" 버튼): api_cut_and_save() -- 고른 높이에서 자르고
         스케일 맞춤 + 정리/스무딩까지 마무리한다.
  (선택, "GNN으로 보내기" 버튼): api_send_to_gnn() -- 완성된 메쉬를 GNN
         추론 서버(다른 PC, `GNN_API_URL`)로 보내 하중 변형 예측 결과를 받는다.

2단계는 3D 렌더링(pyglet)이 들어가서 별도 프로세스(`stage1_orientation_worker.py`)로
매번 새로 띄운다. 나머지 단계는 렌더링이 없어 이 서버 프로세스에서 바로 처리한다.

직접 실행:
    C:/Users/cani0/foot_deform_engine/.venv/Scripts/python.exe webapp/app.py
    -> http://127.0.0.1:5050 접속
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import requests
import trimesh
from flask import Flask, jsonify, render_template, request, send_from_directory

#: 하중 변형 GNN 추론을 맡는 별도 PC의 API 주소.
GNN_API_URL = "http://203.255.175.206:5051/predict"

# ---------------------------------------------------------------------------
# 경로 설정 -- 필요하면 여기만 고치면 됨.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FOOT_ENGINE_SRC = REPO_ROOT / "src"
if str(FOOT_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(FOOT_ENGINE_SRC))

# cp949 등 비-UTF8 콘솔에서 foot_engine 쪽 한글/em-dash 출력이 깨지거나
# UnicodeEncodeError로 죽는 문제 방지 (launcher.py/stage1_orientation_worker.py도 같은 조치).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

# 얼린(frozen) exe엔 별도 python.exe가 없어 "python.exe stage1_orientation_worker.py"
# 식 호출이 안 된다 -- 대신 이 exe 자체를 특수 플래그로 재귀 호출해서 워커
# 역할을 하게 한다(launcher.py의 워커 분기 참고). 개발 모드(python app.py)
# 에서는 그대로 스크립트를 호출한다.
#
# 쓰기 가능해야 하는 경로(_APP_DIR, exe 옆의 jobs/ 폴더)와 읽기 전용 번들
# 자산 경로(_BUNDLE_DIR, PyInstaller가 --add-data로 넣은 templates/,
# data/models/)는 얼린 상태에서 서로 다른 위치라 구분해서 써야 한다.
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

# 얼린(frozen) exe에서는 Flask 기본 template_folder 추정(모듈 위치 기준)이
# PyInstaller 번들 임시 폴더 구조를 못 따라간다 -- exe 옆 templates/를 명시.
app = Flask(__name__, template_folder=str(_BUNDLE_DIR / "templates")) if FROZEN else Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB


def job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise FileNotFoundError(f"알 수 없는 job_id: {job_id}")
    return d


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
    return jsonify(job_id=job_id, filename=input_path.name)


@app.route("/api/stage1_orientation/<job_id>", methods=["POST"])
def api_stage1_orientation(job_id):
    """발을 크롭하고, 발바닥 방향 후보 여러 개를 미리보기 이미지로 만들어 반환한다.

    자동으로 1등 방향만 고르면 가끔 틀릴 수 있어, 사람이 후보 중 눈으로
    보고 고르게 한다. 크롭 결과는 파일로 캐싱해서 -- `api_align_for_cut()`에
    `cropped_file`로 넘기면 재크롭 없이 정렬만 다시 한다.
    """
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
    """마법사 3단계: 고른 방향으로 정렬만 한다(자르기/스케일 전) -- 발목 절단
    슬라이더가 3D로 보여줄 대상. pyglet 렌더링이 없는 순수 CPU 연산이라
    subprocess 격리 없이 바로 처리(크롭 단계와 다름)."""
    try:
        d = job_dir(job_id)
        args = request.get_json(silent=True) or {}
        cropped_file = args.get("cropped_file")
        down_direction = args.get("down_direction")
        if not cropped_file or not down_direction:
            return jsonify(error="cropped_file과 down_direction이 필요합니다"), 400

        mesh = trimesh.load(d / cropped_file, force="mesh", process=False)
        aligned = align_for_manual_cut(mesh, down_direction=np.array(down_direction, dtype=np.float64))

        # 실제 절단(cut_and_save)은 이 고해상도 원본으로 한다.
        out_path = d / "3_aligned_for_cut.glb"
        aligned.export(out_path)

        # 3단계 뷰어는 절단 위치만 보면 되니 사진 텍스처는 필요 없다 -- 정점
        # 수를 줄이고 단색으로 바꾼 가벼운 미리보기를 따로 만들어 로딩을
        # 빠르게 한다. 경계(절단면 테두리)는 정점을 줄이기 전에 먼저 다듬어야
        # 톱니 모양이 덜 도드라진다.
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
    """마법사 4단계("저장" 버튼): 사람이 고른 Y 높이에서 잘라 스케일+정리+
    스무딩까지 한 번에 마친다(pyglet 없음, subprocess 격리 불필요)."""
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
            # 3단계 뷰어의 미세조정 슬라이더가 만든 회전을 각도로 재계산하지 않고
            # three.js가 실제로 쓰는 쿼터니언을 그대로 받아 적용한다 -- Euler 축
            # 순서 컨벤션(XYZ가 Rx·Ry·Rz인지 반대인지)을 직접 맞추려다 뷰어에서
            # 본 것과 저장 결과가 미세하게 어긋나는 위험을 아예 없앤다. trimesh는
            # [w,x,y,z] 순서를 받으므로 재배열.
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

        return jsonify(
            file=out_path.name,
            n_vertices=len(result.mesh.vertices),
            n_faces=len(result.mesh.faces),
            scale_factor=result.scale_factor,
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/send_to_gnn/<job_id>", methods=["POST"])
def api_send_to_gnn(job_id):
    """완성된 메쉬(4단계 결과)를 GNN 추론 서버로 보내 하중 변형 예측 결과를 받는다.

    GNN 서버는 GLB 파일 하나를 받아, 성공하면 예측된 GLB를 그대로,
    실패하면 에러 메시지를 담은 JSON을 돌려준다(`GNN_API_URL` 참고).
    """
    try:
        d = job_dir(job_id)
        src_path = d / "4_final.glb"
        if not src_path.exists():
            return jsonify(error="4단계(완료) 결과가 없습니다"), 400

        with open(src_path, "rb") as f:
            resp = requests.post(
                GNN_API_URL,
                files={"file": (src_path.name, f, "model/gltf-binary")},
                timeout=300,
            )

        content_type = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or "json" in content_type:
            try:
                message = resp.json().get("error", resp.text)
            except ValueError:
                message = resp.text
            return jsonify(error=f"GNN 서버 오류: {message}"), 502

        out_path = d / "5_gnn.glb"
        out_path.write_bytes(resp.content)
        mesh = trimesh.load(out_path, force="mesh", process=False)

        return jsonify(file=out_path.name, n_vertices=len(mesh.vertices), n_faces=len(mesh.faces))
    except requests.exceptions.ConnectionError:
        return jsonify(error=f"GNN 서버({GNN_API_URL})에 연결할 수 없습니다 -- 서버가 켜져 있는지 확인하세요"), 502
    except requests.exceptions.Timeout:
        return jsonify(error="GNN 서버 응답 시간 초과(5분)"), 504
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/jobs/<job_id>/<path:filename>")
def serve_job_file(job_id, filename):
    # model-viewer 미리보기는 inline(기본값)으로 그냥 로드해야 하지만, 다운로드
    # 링크는 <a download>만으로는 브라우저마다 강제 저장이 안 먹는 경우가 있어
    # (Content-Disposition: inline이 우선되는 사례 확인) ?download=1이면
    # attachment로 명시해 확실히 저장 다이얼로그가 뜨게 한다.
    as_attachment = request.args.get("download") == "1"
    return send_from_directory(job_dir(job_id), filename, as_attachment=as_attachment)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
