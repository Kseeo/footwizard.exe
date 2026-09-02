"""메쉬 정렬/절단 유틸 — 발바닥 방향 탐지, 바닥 접지, 수동 절단처럼 사진 없이
메쉬 하나만으로 되는 순수 기하 계산. `stl_foot_extract.postprocess_pipeline`이
이 모듈의 함수들로 마법사 3~4단계(정렬 → 절단 → 접지)를 조립한다.

발목 절단 높이는 자동으로 찾지 않고 사람이 3D 뷰어로 보고 직접 고른다
(`cut_at_height()`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .geometry import pca_axes
from .mesh_postprocess import keep_largest_component

#: 자기신고 발길이 없을 때 쓰는 임시 스케일 기준값(mm). 절대 축척 아님.
DEFAULT_REFERENCE_LENGTH_MM = 250.0


def _fibonacci_sphere(n: int) -> np.ndarray:
    """구 표면에 n개 방향을 고르게 뿌린다((n, 3) 단위벡터, 피보나치 나선)."""
    i = np.arange(n)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = 2.0 * np.pi * i / golden
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _sole_direction_contact_counts(
    surface_points: np.ndarray,
    length_axis: np.ndarray,
    *,
    n_directions: int,
    exclude_cone_deg: float,
    contact_band_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """방향 후보(구면에 고르게 뿌린 것 중 길이축 근처 제외)별 접점 수를 센다.

    `find_sole_direction_candidates()`가 쓰는 핵심 계산. Returns: (directions,
    contact_counts) -- `directions[i]`가 "위(접점 반대쪽)" 방향, 발바닥
    방향은 그 부호를 뒤집은 쪽.
    """
    directions = _fibonacci_sphere(n_directions)
    cos_thresh = np.cos(np.radians(exclude_cone_deg))
    keep = np.abs(directions @ length_axis) <= cos_thresh
    directions = directions[keep]

    diag = float(np.linalg.norm(surface_points.max(axis=0) - surface_points.min(axis=0)))
    band = diag * contact_band_ratio

    proj = surface_points @ directions.T  # (n_sample, n_kept_directions)
    mins = proj.min(axis=0)
    contact_counts = (proj <= (mins + band)).sum(axis=0)
    return directions, contact_counts


@dataclass(slots=True)
class SoleDirectionCandidate:
    """`find_sole_direction_candidates()`가 반환하는 후보 하나."""

    direction: np.ndarray  #: 발바닥(아래) 방향 단위벡터.
    score: int  #: 접점 수(클수록 확신도 높음).


def find_sole_direction_candidates(
    surface_points: np.ndarray,
    length_axis: np.ndarray,
    *,
    k: int = 5,
    n_directions: int = 360,
    exclude_cone_deg: float = 40.0,
    contact_band_ratio: float = 0.03,
    nms_deg: float = 25.0,
) -> list[SoleDirectionCandidate]:
    """접점 수가 많은 방향을 발바닥 후보로, 1등 하나 대신 서로 충분히 떨어진
    (`nms_deg`도 이상) 상위 `k`개를 점수 내림차순으로 반환한다.

    접점 수 1등이 항상 정답은 아니라서(1등과 2등 점수 차이가 좁으면 특히)
    사람이 후보 중 눈으로 보고 고를 수 있도록 여러 개를 준다.
    """
    directions, contact_counts = _sole_direction_contact_counts(
        surface_points, length_axis, n_directions=n_directions,
        exclude_cone_deg=exclude_cone_deg, contact_band_ratio=contact_band_ratio,
    )
    order = np.argsort(-contact_counts)
    cos_nms = np.cos(np.radians(nms_deg))
    chosen: list[int] = []
    for i in order:
        d = directions[i]
        if all(np.dot(d, directions[j]) < cos_nms for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break
    return [SoleDirectionCandidate(direction=-directions[i], score=int(contact_counts[i])) for i in chosen]


def sole_direction_candidates_for_mesh(
    mesh: trimesh.Trimesh,
    *,
    k: int = 5,
    n_surface_samples: int = 20_000,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> list[SoleDirectionCandidate]:
    """`align_sole_down()`이 직접 보는 것과 같은 전처리(`keep_largest_component`)를
    거친 뒤 후보를 계산한다 -- 여기서 받은 방향을 그대로
    `align_sole_down(down_direction=...)`에 넣어도 좌표계(중심점/길이축)가
    어긋나지 않도록 보장하는 용도.
    `kwargs`는 `find_sole_direction_candidates()`로 전달.
    """
    mesh, _, _ = keep_largest_component(mesh)
    centroid = mesh.vertices.mean(axis=0)
    c = mesh.vertices - centroid
    length_axis = pca_axes(c)[:, 0]
    if rng is None:
        rng = np.random.default_rng(0)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples, seed=rng)
    return find_sole_direction_candidates(surface_points - centroid, length_axis, k=k, **kwargs)


def align_sole_down(
    mesh: trimesh.Trimesh,
    *,
    down_direction: np.ndarray,
) -> trimesh.Trimesh:
    """PCA 주축을 좌표축에 맞추고 발바닥 방향으로 X=길이축, Y=높이축(발바닥 -Y),
    Z=너비축으로 맞춘다(중심 원점).

    down_direction: 발바닥 방향(원본 메쉬 좌표계 기준, 단위벡터가 아니어도 됨,
    내부에서 정규화). `sole_direction_candidates_for_mesh()`가 준 후보 중
    사람이 고른 걸 넣는다.
    """
    centroid = mesh.vertices.mean(axis=0)
    c = mesh.vertices - centroid

    length_axis = pca_axes(c)[:, 0]
    down = np.asarray(down_direction, dtype=np.float64)
    down = down / np.linalg.norm(down)

    y_axis = -down
    x_axis = length_axis - (length_axis @ y_axis) * y_axis  # y_axis에 재직교화
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)  # 순수 회전(det=+1) 보장

    final_axes = np.stack([x_axis, y_axis, z_axis], axis=1)

    aligned = mesh.copy()
    aligned.vertices = c @ final_axes
    return aligned


def prune_far_fragments(
    mesh: trimesh.Trimesh,
    *,
    distance_percentile: float = 95.0,
    margin_mult: float = 1.4,
) -> trimesh.Trimesh:
    """`align_sole_down()` 이후(X=길이축, Z=너비축) 발 몸통에서 수평으로 멀리
    떨어진 정점(배경 파편이 얇은 다리로 이어져 위상적으로는 안 떨어지는
    경우)을 잘라내고 다시 최대 연결요소만 남긴다.
    """
    v = mesh.vertices
    center = np.median(v[:, [0, 2]], axis=0)  # X=길이, Z=너비 (align_sole_down 직후 관례)
    dist = np.linalg.norm(v[:, [0, 2]] - center, axis=1)
    threshold = float(np.percentile(dist, distance_percentile)) * margin_mult
    keep = dist <= threshold
    if keep.all():
        return mesh

    face_mask = keep[mesh.faces].all(axis=1)
    out = mesh.copy()
    out.update_faces(face_mask)
    out.remove_unreferenced_vertices()

    out, faces_before, faces_after = keep_largest_component(out)
    if faces_after < faces_before or not keep.all():
        print(
            f"[정리] 발에서 수평으로 멀리 떨어진 파편 제거: 정점 {len(mesh.vertices):,} -> "
            f"{len(out.vertices):,}(거리 임계 {threshold:.4g})"
        )
    return out


def cut_at_height(mesh: trimesh.Trimesh, y: float) -> trimesh.Trimesh:
    """align_sole_down() 직후(Y=높이축) 호출 전제 -- 폭곡선 추정 없이 Y=y에서
    그냥 수평으로 자른다(사람이 3D로 보고 위치를 직접 고르는 UI용).
    """
    trimmed = mesh.slice_plane([0.0, y, 0.0], [0.0, -1.0, 0.0], cap=True)
    if trimmed is None or len(trimmed.vertices) == 0:
        print("[cut] 절단 결과가 비어 원본을 유지합니다")
        return mesh
    trimmed, faces_before, faces_after = keep_largest_component(trimmed)
    if faces_after < faces_before:
        print(f"[cut] 절단 후 부유 조각 제거: 면 {faces_before:,} -> {faces_after:,}")
    print(f"[cut] Y={y:.4g}에서 절단 (정점 {len(mesh.vertices):,} -> {len(trimmed.vertices):,})")
    return trimmed


def rest_on_floor(
    mesh: trimesh.Trimesh,
    *,
    floor_percentile: float = 0.5,
    n_surface_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> trimesh.Trimesh:
    """발바닥이 Y=0에 오도록 Y축으로만 평행이동한다.

    - 단일 최저 정점이 아닌 floor_percentile 백분위 기준(노이즈 스파이크
      하나에 전체 메쉬가 매달리는 문제 방지).
    - 표면적 기준 균등 샘플 사용(정점을 그대로 쓰면 곡률 큰 부위가
      과대표집돼 왜곡됨).
    - align_sole_down()이 정한 좌표계(발바닥=-Y) 전제.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples, seed=rng)
    floor_y = float(np.percentile(surface_points[:, 1], floor_percentile))
    resting = mesh.copy()
    resting.apply_translation([0.0, -floor_y, 0.0])
    return resting


def to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Y=높이 좌표계를 Z=높이로 바꾼다(X축 기준 90도 회전, 형태 왜곡 없음).

    대부분 3D 뷰어/슬라이서는 Z를 위로 가정.
    """
    rotated = mesh.copy()
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    rotated.vertices = np.stack([x, -z, y], axis=1)
    return rotated


def find_floor_contact_mask(
    mesh: trimesh.Trimesh,
    *,
    tolerance_mm: float = 2.0,
    up_axis: int = 2,
) -> np.ndarray:
    """바닥에서 tolerance_mm 이내인 정점을 "접지 노드"로 표시한 불리언 배열
    (정점 순서 1:1 대응)을 반환한다.

    - rest_on_floor() 이후(스케일 완료) 메쉬 전제.
    - up_axis 기본 2(Z) -- `to_z_up()` 적용 전(Y=높이)이면 1로 넘길 것
    """
    heights = mesh.vertices[:, up_axis]
    mask = heights <= (heights.min() + tolerance_mm)
    print(f"[floor-contact] 접지 노드 {int(mask.sum()):,}/{len(mask):,}개 "
          f"({100 * mask.mean():.1f}%, 허용오차 {tolerance_mm:.1f}mm)")
    return mask

