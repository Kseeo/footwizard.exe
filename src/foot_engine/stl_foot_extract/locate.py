"""국소 점 집합이 "발일 가능성"을 점수화하는 형상 휴리스틱 두 개.

`branch_cut.py`가 위상적으로 잘라낸 조각들 중 발 모양에 가장 가까운 걸
고를 때 쓴다(`pick_most_foot_like()`).

- 구형성(sphericity): 발은 국소적으로 통통함, 배경(의자 다리=가늘고 김,
  벽/좌판=넓고 납작)은 다름. 공분산 고유값비(λ3/λ1)로 정량화.
- 다지(多指) 구조(toe cluster): 발가락처럼 한쪽 끝이 여러 둥근 돌기로
  갈라지는 패턴. 국소 주축의 "끝" 쪽 정점을 축 수직 평면에 투영, DBSCAN
  군집 수로 판정.

한계: 경험적 가정이 겹친 휴리스틱이라 100% 신뢰 불가(라운드 쿠션=구형성
오작동, 손 동반 촬영=다지구조 오작동 가능). 최종 판단은 사람이 확인.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def _sphericity(points: np.ndarray) -> float:
    """점 집합의 공분산 고유값비(λ3/λ1, 오름차순 λ3<=λ2<=λ1) -- 0(막대/판)~1(구)."""
    if len(points) < 6:
        return 0.0
    cov = np.cov((points - points.mean(axis=0)).T)
    eigvals = np.linalg.eigvalsh(cov)
    l3, l1 = eigvals[0], eigvals[2]
    if l1 <= 1e-15:
        return 0.0
    return float(l3 / l1)


def _toe_cluster_score(
    points: np.ndarray,
    center: np.ndarray,
    *,
    tip_band_ratio: float = 0.35,
    dbscan_eps_ratio: float = 0.12,
    min_toe_clusters: int = 2,
    max_toe_clusters: int = 7,
) -> float:
    """후보 영역 안에서 "말단이 여러 갈래로 갈라지는" 패턴이 있으면 높은 점수.

    - 영역의 국소 주축(PCA 최장축)에 점들을 투영, 양쪽 끝(tip_band_ratio씩)
      중 다지 구조가 더 뚜렷한 쪽을 택함.
    - 그 끝 쪽 점을 축 수직 평면에 투영해 DBSCAN 군집 수를 셈 --
      min_toe_clusters~max_toe_clusters개면 발가락 패턴(높은 점수), 그 밖
      (매끈한 끝=1개, 노이즈로 과분할)이면 낮은 점수.
    - Returns: 0~1 점수, 다지 구조가 뚜렷할수록 높음.
    """
    if len(points) < 20:
        return 0.0
    c = points - points.mean(axis=0)
    cov = c.T @ c
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # 최대 고유값 -- 국소 주축(길이 방향)

    t = c @ axis
    span = t.max() - t.min()
    if span <= 1e-12:
        return 0.0

    perp_basis = eigvecs[:, :2]  # 축에 수직인 평면(중간/최소 고유값 방향)
    best = 0.0
    for sign in (1.0, -1.0):
        tip_mask = (sign * t) >= (sign * t).max() - span * tip_band_ratio
        tip_pts = c[tip_mask] @ perp_basis
        if len(tip_pts) < 10:
            continue
        eps = float(np.linalg.norm(tip_pts.max(axis=0) - tip_pts.min(axis=0))) * dbscan_eps_ratio
        if eps <= 1e-12:
            continue
        labels = DBSCAN(eps=eps, min_samples=4).fit_predict(tip_pts)
        n_clusters = len(set(labels.tolist()) - {-1})
        if min_toe_clusters <= n_clusters <= max_toe_clusters:
            # 클러스터 수가 이상적인 범위(3~5, 발가락 5개 근방) 중앙에 가까울수록 가점.
            ideal = 4.0
            closeness = 1.0 - min(abs(n_clusters - ideal) / ideal, 1.0)
            best = max(best, 0.6 + 0.4 * closeness)
    return best
