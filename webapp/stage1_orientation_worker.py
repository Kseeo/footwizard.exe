"""마법사 2단계 워커 -- 발을 크롭하고 방향 후보들을 미리보기 이미지로 저장한다.

app.py가 3D 렌더링을 서버 본체와 분리하려고 매 요청마다 별도 프로세스로 띄운다.
결과는 stdout에 JSON 한 줄로 출력(그 외 로그는 stderr).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
sys.stdout = sys.stderr

import trimesh  # noqa: E402

from foot_engine.sfm.dense import align_sole_down, keep_largest_component, sole_direction_candidates_for_mesh  # noqa: E402
from foot_engine.stl_foot_extract.postprocess_pipeline import crop_foot_mesh  # noqa: E402


def _render_preview(mesh: trimesh.Trimesh, out_path: Path) -> None:
    """방향 확인용 빠른 미리보기 한 장(발바닥이 아래로 가게 정렬된 상태에서 옆에서 봄)."""
    scene = mesh.scene()
    scene.set_camera(angles=[0, 0, 0], distance=mesh.scale * 0.9, center=mesh.centroid)
    png = scene.save_image(resolution=(360, 360))
    out_path.write_bytes(png)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_cropped", required=True)
    p.add_argument("--job_dir", required=True)
    # 웹앱이 5개씩 보여주며 "더 보기"로 재크롭 없이 넘기므로 10개를 미리 뽑아둔다.
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--result_json", required=True)
    args = p.parse_args()

    cropped_mesh, _ = crop_foot_mesh(args.input)
    out_cropped = Path(args.output_cropped)
    cropped_mesh.export(out_cropped)

    job_dir = Path(args.job_dir)
    # 후보 계산과 같은 전처리(가장 큰 조각만 남기기)를 거쳐야 좌표계가 맞는다.
    largest, _, _ = keep_largest_component(cropped_mesh)
    candidates = sole_direction_candidates_for_mesh(cropped_mesh, k=args.k)

    info = {"cropped_file": out_cropped.name, "candidates": []}
    for i, cand in enumerate(candidates):
        aligned = align_sole_down(largest, down_direction=cand.direction)
        thumb_name = f"1a_cand{i}.png"
        _render_preview(aligned, job_dir / thumb_name)
        info["candidates"].append({
            "index": i,
            "score": cand.score,
            "direction": cand.direction.tolist(),
            "thumbnail": thumb_name,
        })

    Path(args.result_json).write_text(json.dumps(info), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
