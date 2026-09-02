"""메쉬에서 몸통에 가늘게 이어진 배경 조각을 찾아 잘라낸다.

말단(가지 끝)에서 몸통 쪽으로 걸어 들어가며, 국소 단면 폭이 잘록해졌다가
다시 넓어지는 지점("목")을 찾아 그 자리를 위상적으로 끊는다. 확정 판별기가
아니라 후보 생성기라서, 갈라진 조각들 중 발 모양에 가장 가까운 것을 점수로
골라야 한다(`pick_most_foot_like()`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import dijkstra

from .crop import _remove_vertices
from .locate import _sphericity, _toe_cluster_score


def _geodesic_graph(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """엣지 길이를 가중치로 쓰는 인접 그래프(다익스트라 입력용)."""
    edges = mesh.edges_unique
    v = mesh.vertices
    elen = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    n = len(v)
    return sp.csr_matrix(
        (np.concatenate([elen, elen]),
         (np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n, n),
    )


@dataclass(slots=True)
class BranchTip:
    """말단 정점 하나와, 그 지점에서 전체 정점까지의 지오데식(표면을 따라간) 거리."""

    vertex: int
    dist_from_tip: np.ndarray


def find_branch_tips(
    mesh: trimesh.Trimesh,
    *,
    top_k: int = 6,
    min_separation_ratio: float = 0.15,
) -> list[BranchTip]:
    """서로 멀리 떨어진 말단들을 찾는다(farthest point sampling).

    무게중심에 가장 가까운 정점에서 시작해, 지금까지 찾은 모든 지점으로부터
    가장 먼 정점을 하나씩 추가한다. 새 후보가 기존 지점들과 너무 가까우면
    (`min_separation_ratio`) 멈춘다.
    """
    graph = _geodesic_graph(mesh)
    v = mesh.vertices
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    min_sep = diag * min_separation_ratio

    start = int(np.argmin(np.linalg.norm(v - v.mean(axis=0), axis=1)))
    tips: list[BranchTip] = []
    min_dist_to_chosen = dijkstra(graph, indices=start, directed=False)

    for _ in range(top_k):
        candidate = int(np.argmax(min_dist_to_chosen))
        if tips and min_dist_to_chosen[candidate] < min_sep:
            break
        d = dijkstra(graph, indices=candidate, directed=False)
        tips.append(BranchTip(vertex=candidate, dist_from_tip=d))
        min_dist_to_chosen = np.minimum(min_dist_to_chosen, d)

    return tips


@dataclass(slots=True)
class NeckCut:
    """말단 하나에서 찾은 목(neck) 지점 -- 국소 단면이 좁아졌다 다시 넓어지는 곳."""

    tip_vertex: int
    cut_distance: float
    neck_width: int
    widen_ratio: float


def find_neck_cuts(
    mesh: trimesh.Trimesh,
    tips: list[BranchTip],
    *,
    band_width_edge_mult: float = 3.0,
    widen_ratio: float = 3.0,
    widen_search_bands: int = 5,
    min_neck_vertices: int = 3,
    smoothing_bands: int = 2,
    max_search_ratio: float = 0.5,
) -> list[NeckCut]:
    """각 말단에서 몸통 쪽으로 걸어 들어가며 단면 폭(구간별 정점 수)의 골을 찾는다.

    발가락처럼 끝에서 몸통까지 단조롭게 넓어지는 정상 형태는 걸리지 않고,
    "얇아졌다가(골) 다시 확 넓어지는" 지점만 잡도록, 국소 최솟값이면서 그
    직후 폭이 `widen_ratio`배 이상 넓어지는 조건을 같이 요구한다.
    """
    v = mesh.vertices
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    band_width = float(np.median(edge_len)) * band_width_edge_mult
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_dist = diag * max_search_ratio
    n_bands = max(int(max_dist / band_width), smoothing_bands + widen_search_bands + 2)

    cuts: list[NeckCut] = []
    for tip in tips:
        d = tip.dist_from_tip
        in_range = d < n_bands * band_width
        band_idx = np.clip((d / band_width).astype(int), 0, n_bands - 1)
        counts = np.bincount(band_idx[in_range], minlength=n_bands)[:n_bands].astype(float)
        if smoothing_bands > 1:
            kernel = np.ones(smoothing_bands) / smoothing_bands
            counts = np.convolve(counts, kernel, mode="same")

        for i in range(1, n_bands - widen_search_bands):
            if counts[i] < min_neck_vertices:
                continue
            if not (counts[i] <= counts[i - 1] and counts[i] <= counts[i + 1]):
                continue  # 국소 최솟값(골)이 아님
            after_peak = float(counts[i + 1:i + 1 + widen_search_bands].max())
            ratio = after_peak / counts[i]
            if ratio >= widen_ratio:
                cuts.append(NeckCut(
                    tip_vertex=tip.vertex, cut_distance=i * band_width,
                    neck_width=int(counts[i]), widen_ratio=ratio,
                ))
                break  # 이 말단에서는 몸통에 가장 가까운(첫) 골 하나만

    return cuts


def split_at_necks(
    mesh: trimesh.Trimesh,
    tips: list[BranchTip],
    cuts: list[NeckCut],
    *,
    ring_width_edge_mult: float = 1.5,
) -> trimesh.Trimesh:
    """목 지점마다 그 자리의 얇은 정점 띠를 지워 위상적으로 끊는다.

    실제 조각 분리는 이후 `mesh.split()`(`suggest_neck_components()` 참고)이
    한다 -- 이 함수는 연결만 끊는다.
    """
    if not cuts:
        return mesh

    v = mesh.vertices
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    ring_width = float(np.median(edge_len)) * ring_width_edge_mult

    tip_by_vertex = {t.vertex: t for t in tips}
    remove_mask = np.zeros(len(v), dtype=bool)
    for cut in cuts:
        d = tip_by_vertex[cut.tip_vertex].dist_from_tip
        remove_mask |= np.abs(d - cut.cut_distance) <= ring_width

    if not remove_mask.any():
        return mesh
    return _remove_vertices(mesh, ~remove_mask)


@dataclass(slots=True)
class BendComponent:
    """`suggest_neck_components()`가 반환하는 조각 하나 -- 사람이 고를 후보."""

    mesh: trimesh.Trimesh
    n_vertices: int
    bbox_size: np.ndarray  # (dx, dy, dz)
    sphericity_score: float
    toe_score: float


def pick_most_foot_like(
    components: list[BendComponent],
    *,
    min_size_ratio: float = 0.3,
) -> BendComponent:
    """정점 수 1등을 무조건 고르지 않고, 발 모양 점수(구형성+발가락군집)로 뽑는다.

    배경 조각(예: 의자 다리)이 발보다 커질 수 있어, 크기만으로는 잘못 고를
    수 있다. 가장 큰 조각의 `min_size_ratio` 이상인 후보끼리만 점수로 경쟁시켜,
    너무 작은 조각이 우연히 점수만 높아 이기지 않게 한다.
    """
    if not components:
        raise ValueError("components가 비어 있습니다")
    largest_n = components[0].n_vertices
    eligible = [c for c in components if c.n_vertices >= largest_n * min_size_ratio]
    return max(eligible, key=lambda c: c.sphericity_score + c.toe_score)


def suggest_neck_components(
    mesh: trimesh.Trimesh,
    *,
    tip_top_k: int = 6,
    tip_min_separation_ratio: float = 0.15,
    min_component_vertices: int = 30,
    ring_width_edge_mult: float = 10.0,
    **neck_kwargs,
) -> list[BendComponent]:
    """목 지점(`find_neck_cuts()`)에서 잘라 갈라진 조각들을 크기 내림차순으로 반환한다.

    조각이 하나도 안 갈라지면(목을 못 찾음) 원본 메쉬 하나만 반환한다.
    `neck_kwargs`는 `find_neck_cuts()`로 그대로 전달.
    """
    tips = find_branch_tips(mesh, top_k=tip_top_k, min_separation_ratio=tip_min_separation_ratio)
    cuts = find_neck_cuts(mesh, tips, **neck_kwargs)
    cut_mesh = split_at_necks(mesh, tips, cuts, ring_width_edge_mult=ring_width_edge_mult)

    pieces = cut_mesh.split(only_watertight=False)
    if len(pieces) <= 1:
        pieces = [mesh]

    components: list[BendComponent] = []
    for piece in pieces:
        if len(piece.vertices) < min_component_vertices:
            continue
        pts = piece.vertices
        components.append(BendComponent(
            mesh=piece,
            n_vertices=len(pts),
            bbox_size=piece.bounds[1] - piece.bounds[0],
            sphericity_score=_sphericity(pts),
            toe_score=_toe_cluster_score(pts, pts.mean(axis=0)),
        ))
    components.sort(key=lambda c: c.n_vertices, reverse=True)
    return components
