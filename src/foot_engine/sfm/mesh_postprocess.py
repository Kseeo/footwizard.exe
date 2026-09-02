"""메쉬 정리 유틸 두 개 -- `dense.py`가 정렬/마감 단계에서 재사용한다.
"""

from __future__ import annotations

import numpy as np
import trimesh


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, int]:
    """면(face) 기준 가장 큰 연결 요소만 남기고 부유 파편을 지운다.
    - 공간적으로 분리된 덩어리 단위로만 자른다.
    - Returns: (필터링된 메쉬, 원본 face 수, 남은 face 수).
    """
    total_faces = len(mesh.faces)
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, total_faces, total_faces
    largest = max(components, key=lambda c: len(c.faces))
    return largest, total_faces, len(largest.faces)


def finish_smooth_mesh(
    mesh: trimesh.Trimesh,
    *,
    lamb: float = 0.5,
    iterations: int = 40,
) -> trimesh.Trimesh:
    """전체 정점에 평범한(가중치 없는) 라플라시안 스무딩을 반복해 남은 고주파
    표면 노이즈를 마감 처리한다.

    Args:
        lamb: 반복당 이웃 평균 쪽으로 당기는 비율(0~1).
        iterations: 반복 횟수 -- 클수록 매끈해지지만 디테일도 더 죽는다.
    """
    out = mesh.copy()
    trimesh.smoothing.filter_laplacian(out, lamb=lamb, iterations=iterations, volume_constraint=False)
    if not np.isfinite(out.vertices).all():
        print("[postprocess][경고] 마감 스무딩 결과에 비정상 값(NaN/Inf)이 생겨 이번 단계는 건너뜁니다")
        return mesh
    print(f"[postprocess] 마감 스무딩(라플라시안 x{iterations}): 정점 {len(out.vertices):,}개")
    return out
