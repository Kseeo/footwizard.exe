"""메쉬 정렬/절단 유틸 패키지 -- `stl_foot_extract`가 재사용하는 순수 기하 계산 모음.

    dense.align_sole_down() — PCA 축 정렬 + 발바닥 방향 탐지
    dense.cut_at_height() — 사람이 고른 높이에서 수동 절단
    dense.decimate_mesh() — 정점 수를 목표치까지 줄임
    dense.rest_on_floor() / to_z_up() — 좌표계 정리
    masking.load_skin_segmenter() / skin_only_mask() — 피부 분할 (texture_crop.py가 씀)

`geometry.py`는 스케일 추정용 유틸.
"""

from __future__ import annotations

from .dense import (
    DEFAULT_REFERENCE_LENGTH_MM,
    DEFAULT_TARGET_VERTICES,
    SoleDirectionCandidate,
    align_sole_down,
    cut_at_height,
    decimate_mesh,
    find_floor_contact_mask,
    find_sole_direction_candidates,
    prune_far_fragments,
    rest_on_floor,
    sole_direction_candidates_for_mesh,
    to_z_up,
)
from .geometry import measured_length, pca_axes
from .masking import load_skin_segmenter, skin_only_mask
from .mesh_postprocess import finish_smooth_mesh, keep_largest_component

__all__ = [
    # masking
    "load_skin_segmenter",
    "skin_only_mask",
    # geometry
    "measured_length",
    "pca_axes",
    # mesh_postprocess
    "keep_largest_component",
    "finish_smooth_mesh",
    # dense (메쉬 정렬/절단)
    "DEFAULT_REFERENCE_LENGTH_MM",
    "DEFAULT_TARGET_VERTICES",
    "SoleDirectionCandidate",
    "align_sole_down",
    "cut_at_height",
    "decimate_mesh",
    "find_floor_contact_mask",
    "find_sole_direction_candidates",
    "prune_far_fragments",
    "rest_on_floor",
    "sole_direction_candidates_for_mesh",
    "to_z_up",
]
