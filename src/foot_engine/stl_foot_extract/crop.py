"""메쉬 정점 부분집합을 안전하게 잘라내는 저수준 유틸.

`branch_cut.py`(위상 절단)와 `texture_crop.py`(피부투표 크롭)가 공통으로 쓴다.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _remove_vertices(mesh: trimesh.Trimesh, keep_mask: np.ndarray) -> trimesh.Trimesh:
    """정점 부분집합만 남기고 나머지를 잘라낸다.

    Args:
        mesh: 원본 메쉬.
        keep_mask: 남길 정점만 True인 불리언 배열(mesh.vertices와 1:1 대응).
    """
    face_mask = keep_mask[mesh.faces].all(axis=1)
    out = mesh.copy()
    out.update_faces(face_mask)
    out.remove_unreferenced_vertices()
    return out
