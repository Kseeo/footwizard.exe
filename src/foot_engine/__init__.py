"""foot_engine — 3D 발 스캔(STL/GLB) 추출/정리 파이프라인.

Quick start::

    from foot_engine.stl_foot_extract import crop_foot_mesh, align_for_manual_cut, cut_and_finish_mesh

두 서브패키지:
    stl_foot_extract/ — 실제 진입점(웹앱 마법사가 쓰는 경로). 스캔 원본에서
        발 부위 크롭+정렬+수동 절단+정리.
    sfm/ — stl_foot_extract가 재사용하는 메쉬 정렬/절단 유틸(`sfm/__init__.py` 참고).
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
