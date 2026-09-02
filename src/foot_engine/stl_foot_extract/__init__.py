"""색/텍스처가 있는 3D 스캔(GLB 등, 배경/다른 물체가 같이 찍힌 경우)에서 발
부위를 찾아 잘라내고 다듬는 패키지/ 입력은 메쉬 파일(GLB 등) 

하위 모듈:
    texture_crop.py -- 다중 뷰 피부분할 투표로 발 부위만 크롭
    branch_cut.py   -- 잘록해졌다 다시 넓어지는 지점을 찾아 배경 조각 분리
    crop.py         -- 정점 부분집합을 안전하게 잘라내는 저수준 유틸
    finishing.py    -- 배경 파편 정리 + 스무딩(구멍 메움/사포질/고곡률 완화)
    postprocess_pipeline.py -- 마법사가 직접 부르는 진입점: `crop_foot_mesh()`
        (2단계, 방향 후보 계산용 크롭) / `align_for_manual_cut()`(3단계, 정렬)
        / `cut_and_finish_mesh()`(4단계, 절단+스케일+정리)

사용 예::

    from foot_engine.stl_foot_extract import crop_foot_mesh, cut_and_finish_mesh, align_for_manual_cut

    cropped, _ = crop_foot_mesh("scan.glb")
    aligned = align_for_manual_cut(cropped, down_direction=[0, -1, 0])
    result = cut_and_finish_mesh(aligned, cut_y=0.3)
    result.mesh.export("scan_foot.glb")
"""

from __future__ import annotations

from .finishing import finish_smooth_mesh, keep_largest_component, postprocess_mesh, smooth_boundary_loops
from .postprocess_pipeline import (
    FootPipelineResult,
    align_for_manual_cut,
    crop_foot_mesh,
    cut_and_finish_mesh,
)

__all__ = [
    "FootPipelineResult",
    "crop_foot_mesh",
    "align_for_manual_cut",
    "cut_and_finish_mesh",
    "keep_largest_component",
    "finish_smooth_mesh",
    "smooth_boundary_loops",
    "postprocess_mesh",
]
