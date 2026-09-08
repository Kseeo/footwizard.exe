"""GLB(색/텍스처 있는 원본) 하나로 "발 검출 → 정렬 → 발목 절단(수동) →
정리/스무딩 → 접지 노드 계산"까지 잇는, 웹앱 마법사(`webapp/app.py`,
`stage1_orientation_worker.py`)가 직접 호출하는 함수들.

마법사 흐름(각 단계 구현은 다른 모듈에 있음, 여기는 순서만 고정):
    1. `crop_foot_mesh()` -- `texture_crop.extract_by_skin_vote()`로 다중뷰
       피부분할 투표해 발 부위만 크롭 (`stage1_orientation_worker.py`가 호출)
    2. `align_for_manual_cut()` -- 고른 방향으로 정렬만(자르기 전)
    3. `cut_and_finish_mesh()` -- 사람이 고른 높이에서 절단 + `finishing.
       postprocess_mesh()`(배경 파편 제거+스무딩+구멍 메움) + 스케일 +
       `sfm.dense.decimate_mesh()`(정점 수 맞춤) +
       `sfm.dense.find_floor_contact_mask()`(접지 노드)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from foot_engine.sfm.dense import (
    DEFAULT_REFERENCE_LENGTH_MM,
    DEFAULT_TARGET_VERTICES,
    align_sole_down,
    cut_at_height,
    decimate_mesh,
    find_floor_contact_mask,
    keep_largest_component,
    prune_far_fragments,
    rest_on_floor,
    to_z_up,
)
from foot_engine.sfm.geometry import measured_length

from .branch_cut import pick_most_foot_like, suggest_neck_components
from .finishing import postprocess_mesh
from .texture_crop import extract_by_skin_vote, load_textured_mesh


@dataclass(slots=True)
class FootPipelineResult:
    """`cut_and_finish_mesh()` 결과."""

    mesh: trimesh.Trimesh
    n_input_vertices: int
    scale_factor: float  #: 원본 대비 적용된 스케일 배율.
    up_axis: str  #: "Z" 또는 "Y" -- `floor_contact_mask`가 어느 축 기준인지.
    floor_contact_mask: np.ndarray | None  #: `mesh.vertices`와 1:1 대응하는 불리언 배열.


def crop_foot_mesh(
    mesh_path: str | Path,
    *,
    n_views: int = 16,
    resolution: int = 640,
    vote_threshold: float = 0.5,
    close_gap_radius_mult: float = 25.0,
) -> tuple[trimesh.Trimesh, int]:
    """색/텍스처가 있는 메쉬 파일에서 발 부위만 찾아 잘라낸다.

    `texture_crop.extract_by_skin_vote()`를 감싼 것 -- 인자 설명은 그쪽
    docstring 참고. 마법사 2단계(방향 후보 계산, `stage1_orientation_worker.py`)가
    부른다.

    Args:
        mesh_path: 입력 메쉬 파일 경로(GLB 등).
        n_views: 가상 카메라로 렌더할 시점 개수.
        resolution: 렌더 해상도(가로 픽셀, 세로는 자동으로 0.75배).
        vote_threshold: 관측된 뷰 중 이 비율 이상 피부로 보인 정점만 채택.
        close_gap_radius_mult: 작은 빈틈을 메우는 반경(전형적 정점 간격의 배수).

    Returns:
        (크롭된 메쉬, 입력 메쉬 정점 수)
    """
    mesh_path = Path(mesh_path)
    mesh = load_textured_mesh(mesh_path)
    n_input = len(mesh.vertices)
    print(f"[foot-pipeline] 입력: {mesh_path} (정점 {n_input:,}개)")

    result = extract_by_skin_vote(
        mesh, n_views=n_views, resolution=(resolution, int(resolution * 0.75)),
        vote_threshold=vote_threshold, close_gap_radius_mult=close_gap_radius_mult,
    )
    out_mesh = result.mesh

    return out_mesh, n_input


def align_for_manual_cut(
    cropped_mesh: trimesh.Trimesh,
    *,
    down_direction: np.ndarray,
) -> trimesh.Trimesh:
    """마법사 3단계(절단 위치): 배경 파편을 지우고 발바닥 방향으로 정렬만
    한다(자르기/스케일은 다음 단계).

    Args:
        cropped_mesh: `crop_foot_mesh()`가 만든 크롭 결과.
        down_direction: 발바닥 방향(사람이 2단계에서 고른 후보).
    """
    mesh, _, _ = keep_largest_component(cropped_mesh)
    return align_sole_down(mesh, down_direction=down_direction)


def cut_and_finish_mesh(
    aligned_mesh: trimesh.Trimesh,
    *,
    cut_y: float,
    n_input_vertices: int = 0,
    reference_length_mm: float | None = None,
    z_up: bool = True,
    prune_neck_fragments: bool = True,
    prune_neck_fragments_kwargs: dict | None = None,
    postprocess: bool = True,
    sand_iterations: int = 3,
    curvature_iterations: int = 150,
    finish_smooth_iterations: int = 10,
    fill_holes_max_diameter_ratio: float = 0.05,
    fill_round_holes_enabled: bool = True,
    fill_round_holes_min_circularity: float = 0.5,
    floor_contact_tolerance_mm: float | None = 2.0,
    target_vertices: int | None = DEFAULT_TARGET_VERTICES,
    decimate_smooth_after: bool = True,
) -> FootPipelineResult:
    """마법사 4단계("저장" 버튼): `align_for_manual_cut()`이 만든 메쉬를
    사람이 고른 `cut_y` 높이에서 자르고 스케일+정리+스무딩+해상도 맞춤까지
    마무리한다.

    스무딩은 절단 전에 하고(먼저 표면을 매끈하게 만들면 절단면 자체가
    깨끗하게 나옴), 스케일은 절단 후에 계산한다(절단 전 길이엔 다리까지
    포함돼 있어 발 길이 기준으로 못 씀).

    Args:
        aligned_mesh: `align_for_manual_cut()`의 결과.
        cut_y: 사람이 고른 절단 높이(Y).
        reference_length_mm: 자기신고 발 길이(mm). 없으면 임시 기준값 사용.
        postprocess: 절단 전 스무딩(구멍 메움/사포질/고곡률 완화)을 할지.
        prune_neck_fragments: 정렬 후 잘록해졌다 넓어지는 지점에서 배경
            파편을 잘라낼지.
        floor_contact_tolerance_mm: 지정하면 바닥에서 이 거리(mm) 이내
            정점을 표시하는 마스크를 같이 계산한다.
        target_vertices: 최종 정점 수를 이 값 근방으로 맞춘다(`decimate_mesh()`).
            None이면 축약하지 않는다.
        decimate_smooth_after: 축약 직후 마감 스무딩을 할지.
    """
    mesh = aligned_mesh
    if postprocess:
        mesh, _ = postprocess_mesh(
            mesh,
            sand_iterations=sand_iterations,
            curvature_iterations=curvature_iterations,
            finish_smooth_iterations=finish_smooth_iterations,
            fill_holes_max_diameter_ratio=fill_holes_max_diameter_ratio,
            fill_round_holes_enabled=fill_round_holes_enabled,
            fill_round_holes_min_circularity=fill_round_holes_min_circularity,
        )

    mesh = cut_at_height(mesh, cut_y)
    mesh = prune_far_fragments(mesh)

    resolved_reference_length_mm = reference_length_mm
    if resolved_reference_length_mm is None:
        resolved_reference_length_mm = DEFAULT_REFERENCE_LENGTH_MM
        print(
            f"[스케일] 자기신고 발길이 없음 — placeholder {DEFAULT_REFERENCE_LENGTH_MM:.0f}mm 기준으로"
            " 스케일링(절대 축척 아님, 형태 비교/시각화용 임시값 — 실사용 전 반드시 확인할 것)"
        )
    own_length = measured_length(mesh.vertices)
    scale_factor = resolved_reference_length_mm / own_length
    mesh.apply_scale(scale_factor)
    print(
        f"[스케일] 메쉬 자체 PCA 길이 {own_length:.4f}(입력 메쉬 임의 단위) -> "
        f"{resolved_reference_length_mm:.1f}mm 기준(x{scale_factor:.4f})"
    )

    if prune_neck_fragments:
        n_before = len(mesh.vertices)
        components = suggest_neck_components(mesh, **(prune_neck_fragments_kwargs or {}))
        if len(components) > 1:
            chosen = pick_most_foot_like(components)
            mesh = chosen.mesh
            print(
                f"[cut] 목(neck) 감지로 파편 조각 분리: {len(components)}개 중 "
                f"발 모양 점수로 채택(정점 {n_before:,} -> {len(mesh.vertices):,})"
            )

    if target_vertices is not None:
        mesh = decimate_mesh(mesh, target_vertices=target_vertices, smooth_after=decimate_smooth_after)

    mesh = rest_on_floor(mesh)

    floor_contact_mask: np.ndarray | None = None
    if floor_contact_tolerance_mm is not None:
        floor_contact_mask = find_floor_contact_mask(mesh, tolerance_mm=floor_contact_tolerance_mm, up_axis=1)

    if z_up:
        mesh = to_z_up(mesh)

    return FootPipelineResult(
        mesh=mesh,
        n_input_vertices=n_input_vertices,
        scale_factor=scale_factor,
        up_axis="Z" if z_up else "Y",
        floor_contact_mask=floor_contact_mask,
    )

