"""배경/파편 제거 + 스무딩(구멍 메움/사포질/고곡률 완화/마감) -- 입력이 메쉬
하나뿐, 사진/카메라 정보 불필요. `postprocess_pipeline.py`가 이 모듈의
`postprocess_mesh()`를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import dijkstra

#: `smooth_high_curvature_regions()` 기본 강도(곡률 백분위 임계값).
DEFAULT_CURVATURE_PERCENTILE = 60.0
DEFAULT_CURVATURE_MIN_RADIUS_MULT = 2.0
DEFAULT_CURVATURE_MAX_RADIUS_MULT = 25.0
DEFAULT_CURVATURE_ITERATIONS = 150
DEFAULT_CURVATURE_ALPHA = 0.7
DEFAULT_CURVATURE_MU = -0.75


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, int]:
    """면(face) 기준 가장 큰 연결 요소만 남기고 부유 파편을 지운다.

    이미 공간적으로 분리된 덩어리 단위로만 자르므로 안전하다 -- 다만 발
    표면에 이어져(fused) 붙은 경계 노이즈는 같은 덩어리라 이걸로 안 떨어진다.

    Returns:
        (필터링된 메쉬, 원본 face 수, 남은 face 수).
    """
    total_faces = len(mesh.faces)
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, total_faces, total_faces
    largest = max(components, key=lambda c: len(c.faces))
    return largest, total_faces, len(largest.faces)


def fill_small_holes(
    mesh: trimesh.Trimesh,
    *,
    max_hole_diameter_ratio: float = 0.05,
) -> tuple[trimesh.Trimesh, int]:
    """작은 구멍(핀홀)만 중심점-팬으로 메운다. 큰 구멍(발바닥 등)은 그대로 둔다.

    Args:
        max_hole_diameter_ratio: 구멍 바운딩박스 대각선 / 메쉬 전체 대각선 비율 상한.

    Returns:
        (구멍 메운 메쉬, 실제로 메운 구멍 개수).
    """
    if len(mesh.faces) < 3 or mesh.is_watertight:
        return mesh, 0

    boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        return mesh, 0

    boundary = mesh.edges[boundary_groups]
    holes = nx.cycle_basis(nx.from_edgelist(boundary))
    if not holes:
        return mesh, 0

    mesh_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_diag = mesh_diag * max_hole_diameter_ratio
    small_holes = []
    for loop in holes:
        pts = mesh.vertices[loop]
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if diag <= max_diag:
            small_holes.append(loop)
    if not small_holes:
        return mesh, 0

    out = _fill_holes_centroid_fan(mesh, small_holes, boundary)
    return out, len(small_holes)


def _fill_holes_centroid_fan(
    mesh: trimesh.Trimesh, holes: list[list[int]], boundary: np.ndarray,
) -> trimesh.Trimesh:
    """구멍마다 새 중심점 정점을 하나씩 추가하고 그 중심에서 부채꼴로 채운다.

    경계의 한 귀퉁이에서 부채꼴을 펴는 방식과 달리, 모든 삼각형이 중심 한
    점을 공유해서 구멍 모양이 볼록하지 않아도 찌그러지지 않는다.

    Args:
        mesh: 원본 메쉬.
        holes: 메울 구멍마다 그 경계를 이루는 정점 인덱스 리스트.
        boundary: 원본 경계 엣지 전체 -- 새로 만든 면의 방향(winding)을
            기존 메쉬와 맞추는 데 쓴다.
    """
    n_existing = len(mesh.vertices)
    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []
    for loop in holes:
        centroid = mesh.vertices[loop].mean(axis=0)
        center_idx = n_existing + len(new_vertices)
        new_vertices.append(centroid)
        n = len(loop)
        for i in range(n):
            a, b = loop[i], loop[(i + 1) % n]
            new_faces.append([center_idx, a, b])
    if not new_faces:
        return mesh

    new_faces_arr = np.array(new_faces, dtype=np.int64)
    new_edges = trimesh.geometry.faces_to_edges(new_faces_arr)
    hashable_new = trimesh.grouping.hashable_rows(new_edges)
    hashable_old = trimesh.grouping.hashable_rows(boundary)
    needs_reverse = np.isin(hashable_new, hashable_old).reshape((-1, 3)).any(axis=1)
    new_faces_arr[needs_reverse] = np.fliplr(new_faces_arr[needs_reverse])

    out = mesh.copy()
    out.vertices = np.vstack([out.vertices, np.array(new_vertices)])
    out.faces = np.vstack([out.faces, new_faces_arr])
    return out


def _hole_diag_and_circularity(mesh: trimesh.Trimesh, loop: list[int]) -> tuple[float, float]:
    """구멍 경계 루프 하나의 (bbox 대각선, 원형도) -- 원형도는 4π·면적/둘레², 원이면 1.0."""
    pts = mesh.vertices[loop]
    centroid = pts.mean(axis=0)
    c = pts - centroid
    _, evecs = np.linalg.eigh(c.T @ c)
    pts2d = c @ evecs[:, 1:]  # 최소 고유값(법선) 축 제외한 평면 2축에 투영
    x, y = pts2d[:, 0], pts2d[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    perimeter = float(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum())
    circularity = 4 * np.pi * area / (perimeter**2 + 1e-12)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return diag, circularity


def fill_round_holes(
    mesh: trimesh.Trimesh,
    *,
    min_circularity: float = 0.7,
    max_hole_diameter_ratio: float = 0.6,
) -> tuple[trimesh.Trimesh, int]:
    """둥근 구멍(단순 미관측 결손으로 추정)만 크기와 무관하게 메운다.

    `fill_small_holes()`는 크기만 보므로 발목 절단면처럼 원래 열려있어야
    할 큰 구멍과, 그냥 못 찍힌 큰 구멍을 구분하지 못한다. 이 함수는 대신
    구멍의 원형도(둥근 정도)로 둘을 구분한다 -- 절단면은 길쭉하고 각져
    원형도가 낮고, 단순 미관측 구멍은 대체로 둥글다.

    Args:
        min_circularity: 이 값 이상 둥근 구멍만 메운다(1.0 = 완전한 원).
        max_hole_diameter_ratio: 메쉬 전체 대각선 대비, 이보다 큰 구멍은
            원형도와 무관하게 메우지 않는다(안전판).
    """
    if len(mesh.faces) < 3 or mesh.is_watertight:
        return mesh, 0

    boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        return mesh, 0

    boundary = mesh.edges[boundary_groups]
    holes = nx.cycle_basis(nx.from_edgelist(boundary))
    if not holes:
        return mesh, 0

    mesh_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_diag = mesh_diag * max_hole_diameter_ratio
    round_holes = []
    for loop in holes:
        diag, circularity = _hole_diag_and_circularity(mesh, loop)
        if diag <= max_diag and circularity >= min_circularity:
            round_holes.append(loop)
    if not round_holes:
        return mesh, 0

    out = _fill_holes_centroid_fan(mesh, round_holes, boundary)
    return out, len(round_holes)


def smooth_boundary_loops(
    mesh: trimesh.Trimesh, *, iterations: int = 15, lamb: float = 0.5,
) -> trimesh.Trimesh:
    """열린 경계(발목 절단면 등)의 테두리만 다듬어 톱니 모양을 완화한다.

    경계 정점끼리만 이웃 평균을 내며, 표면 안쪽 정점은 건드리지 않는다.

    Args:
        iterations: 반복 횟수.
        lamb: 반복당 이웃 평균 쪽으로 당기는 비율(0~1).
    """
    boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        return mesh
    boundary = mesh.edges[boundary_groups]

    adj: dict[int, set[int]] = {}
    for a, b in boundary:
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))

    v = mesh.vertices.copy()
    idxs = np.array(list(adj.keys()))
    for _ in range(iterations):
        new_v = v.copy()
        for i in idxs:
            neighbors = list(adj[i])
            avg = v[neighbors].mean(axis=0)
            new_v[i] = v[i] + lamb * (avg - v[i])
        v = new_v

    out = mesh.copy()
    out.vertices = v
    return out


def _ring_neighbors_padded(
    adjacency: list, *, min_neighbors: int, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """위상 인접(1-ring)을 BFS로 확장해 정점별 이웃 최소 개수를 채운다."""
    n = len(adjacency)
    idx_padded = np.zeros((n, max_neighbors), dtype=np.int64)
    mask = np.zeros((n, max_neighbors), dtype=bool)
    for i in range(n):
        visited = {i}
        frontier = {i}
        neighbors: set[int] = set()
        while len(neighbors) < min_neighbors and frontier:
            next_frontier = {nb for node in frontier for nb in adjacency[node] if nb not in visited}
            visited |= next_frontier
            neighbors |= next_frontier
            frontier = next_frontier
        chosen = np.fromiter(neighbors, dtype=np.int64, count=len(neighbors))[:max_neighbors]
        idx_padded[i, : len(chosen)] = chosen
        mask[i, : len(chosen)] = True
    return idx_padded, mask


def sand_surface(
    mesh: trimesh.Trimesh,
    *,
    min_neighbors: int = 16,
    max_neighbors: int = 32,
    iterations: int = 3,
    regularization: float = 1e-6,
    max_offset_ratio: float = 1.0,
) -> trimesh.Trimesh:
    """모든 정점을 국소 이차곡면(quadric) 근사에 투영해 다듬는다("사포질").

    법선 방향으로만 움직이므로 접평면 방향의 진짜 형태는 보존하면서 그
    정점만 튀는 고주파 노이즈를 깎아낸다. 정점/면 개수는 그대로다.
    """
    n = len(mesh.vertices)
    if n < max(min_neighbors, 6) + 1:
        return mesh

    idx, mask = _ring_neighbors_padded(
        mesh.vertex_neighbors, min_neighbors=min_neighbors, max_neighbors=max_neighbors
    )
    valid = mask.sum(axis=1) >= 6
    maskf = mask.astype(np.float64)

    v = mesh.vertices.copy().astype(np.float64)
    for _ in range(iterations):
        rel = (v[idx] - v[:, None, :]) * maskf[:, :, None]
        cov = np.einsum("nki,nkj->nij", rel, rel)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        u_axis, v_axis = evecs[:, :, 2], evecs[:, :, 1]

        dist = np.linalg.norm(rel, axis=2)
        scale = np.where(valid, dist.sum(axis=1) / np.maximum(mask.sum(axis=1), 1), 1.0)
        scale = np.maximum(scale, 1e-9)

        u = (np.einsum("nki,ni->nk", rel, u_axis) / scale[:, None]) * maskf
        w = (np.einsum("nki,ni->nk", rel, v_axis) / scale[:, None]) * maskf
        h = np.einsum("nki,ni->nk", rel, normal) * maskf

        design = np.stack([u * u, u * w, w * w, u, w, maskf], axis=-1)
        AtA = np.einsum("nki,nkj->nij", design, design)
        AtA += regularization * min_neighbors * np.eye(6)
        Ath = np.einsum("nki,nk->ni", design, h)
        coeffs = np.linalg.solve(AtA, Ath[..., None])[..., 0]

        offset = np.where(valid, coeffs[:, 5], 0.0)
        max_offset = max_offset_ratio * scale
        offset = np.clip(offset, -max_offset, max_offset)
        v = v + offset[:, None] * normal

    out = mesh.copy()
    out.vertices = v
    print(f"[finishing] 사포질(전체 이차곡면 투영): 정점 {n:,}개, {iterations}회 반복")
    return out


def smooth_high_curvature_regions(
    mesh: trimesh.Trimesh,
    *,
    curvature_percentile: float = 90.0,
    min_radius_edge_mult: float = 2.0,
    max_radius_edge_mult: float = 25.0,
    iterations: int = 10,
    alpha: float = 0.6,
    mu: float = -0.65,
) -> trimesh.Trimesh:
    """곡률이 튀는 영역을 Taubin(λ|μ) 스무딩한다. 나머지 정점은 그대로 둔다.

    코어 영역(연결된 고곡률 정점 덩어리)마다 그 영역의 곡률 반경(1/|곡률|)에
    비례해서 확산 반경을 다르게 준다(curvature-adaptive smoothing).
    """
    v = mesh.vertices.copy()
    n = len(v)
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    typical_edge = float(np.median(edge_len))
    curv = trimesh.curvature.discrete_mean_curvature_measure(mesh, v, typical_edge * 4)

    thresh = np.percentile(np.abs(curv), curvature_percentile)
    core_mask = np.abs(curv) > thresh

    neighbors = mesh.vertex_neighbors
    weight = np.zeros(n)

    if core_mask.any():
        visited = np.zeros(n, dtype=bool)
        clusters: list[np.ndarray] = []
        for start in np.where(core_mask)[0]:
            if visited[start]:
                continue
            stack, comp = [start], [start]
            visited[start] = True
            while stack:
                cur = stack.pop()
                for nb in neighbors[cur]:
                    if core_mask[nb] and not visited[nb]:
                        visited[nb] = True
                        comp.append(nb)
                        stack.append(nb)
            clusters.append(np.array(comp, dtype=np.int64))

        edges_u = mesh.edges_unique
        elen_u = np.linalg.norm(v[edges_u[:, 0]] - v[edges_u[:, 1]], axis=1)
        graph = sp.csr_matrix(
            (np.concatenate([elen_u, elen_u]),
             (np.concatenate([edges_u[:, 0], edges_u[:, 1]]), np.concatenate([edges_u[:, 1], edges_u[:, 0]]))),
            shape=(n, n),
        )
        min_radius = typical_edge * min_radius_edge_mult
        max_radius = typical_edge * max_radius_edge_mult

        for comp in clusters:
            local_radius = 1.0 / np.maximum(np.abs(curv[comp]), 1e-9)
            spread_radius = float(np.clip(np.median(local_radius), min_radius, max_radius))
            dist = dijkstra(graph, indices=comp, min_only=True, limit=spread_radius, directed=False)
            reached = np.isfinite(dist)
            w = np.clip(1.0 - dist[reached] / spread_radius, 0.0, 1.0)
            weight[reached] = np.maximum(weight[reached], w)
        weight[core_mask] = 1.0

    rows = np.concatenate([np.full(len(neighbors[i]), i) for i in range(n)])
    cols = np.concatenate([np.asarray(neighbors[i], dtype=np.int64) for i in range(n)])
    deg = np.array([len(neighbors[i]) for i in range(n)])
    vals = np.concatenate([np.full(len(neighbors[i]), 1.0 / len(neighbors[i])) if len(neighbors[i]) else np.array([])
                            for i in range(n)])
    adj = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    isolated = deg == 0
    if isolated.any():
        adj = adj + sp.csr_matrix((np.ones(isolated.sum()), (np.where(isolated)[0], np.where(isolated)[0])), shape=(n, n))

    def laplacian_step(vv: np.ndarray, step: float) -> np.ndarray:
        avg = adj @ vv
        return vv + weight[:, None] * step * (avg - vv)

    new_v = v.copy()
    for _ in range(iterations):
        new_v = laplacian_step(new_v, alpha)
        new_v = laplacian_step(new_v, mu)

    out = mesh.copy()
    out.vertices = new_v
    print(
        f"[finishing] 고곡률 국소 스무딩(Taubin λ|μ): 정점 {int((weight > 0).sum()):,}/{n:,}개 영향"
        f"(코어 {int(core_mask.sum()):,}개, curvature_percentile={curvature_percentile})"
    )
    return out


def finish_smooth_mesh(
    mesh: trimesh.Trimesh,
    *,
    lamb: float = 0.5,
    iterations: int = 40,
) -> trimesh.Trimesh:
    """전체 정점에 라플라시안 스무딩을 반복해 남은 고주파 표면 노이즈를
    마감 처리한다.

    Args:
        lamb: 반복당 이웃 평균 쪽으로 당기는 비율(0~1).
        iterations: 반복 횟수 -- 클수록 매끈해지지만 디테일도 더 죽는다.
    """
    out = mesh.copy()
    trimesh.smoothing.filter_laplacian(out, lamb=lamb, iterations=iterations, volume_constraint=False)
    if not np.isfinite(out.vertices).all():
        print("[finishing][경고] 마감 스무딩 결과에 비정상 값(NaN/Inf)이 생겨 이번 단계는 건너뜁니다")
        return mesh
    print(f"[finishing] 마감 스무딩(라플라시안 x{iterations}): 정점 {len(out.vertices):,}개")
    return out


@dataclass(slots=True)
class PostprocessStats:
    """`postprocess_mesh()`가 남기는 단계별 요약."""

    faces_before: int = 0
    faces_after_largest_component: int = 0
    n_holes_filled: int = 0
    n_round_holes_filled: int = 0
    steps_applied: list[str] = field(default_factory=list)


def postprocess_mesh(
    mesh: trimesh.Trimesh,
    *,
    keep_largest: bool = True,
    fill_holes: bool = True,
    fill_holes_max_diameter_ratio: float = 0.05,
    fill_round_holes_enabled: bool = False,
    fill_round_holes_min_circularity: float = 0.7,
    fill_round_holes_max_diameter_ratio: float = 0.6,
    smooth_boundary_enabled: bool = True,
    smooth_boundary_iterations: int = 15,
    smooth_boundary_lambda: float = 0.5,
    sand_surface_enabled: bool = True,
    sand_min_neighbors: int = 16,
    sand_max_neighbors: int = 32,
    sand_iterations: int = 3,
    smooth_high_curvature: bool = True,
    curvature_percentile: float = DEFAULT_CURVATURE_PERCENTILE,
    curvature_min_radius_mult: float = DEFAULT_CURVATURE_MIN_RADIUS_MULT,
    curvature_max_radius_mult: float = DEFAULT_CURVATURE_MAX_RADIUS_MULT,
    curvature_iterations: int = DEFAULT_CURVATURE_ITERATIONS,
    curvature_alpha: float = DEFAULT_CURVATURE_ALPHA,
    curvature_mu: float = DEFAULT_CURVATURE_MU,
    finish_smooth: bool = True,
    finish_smooth_lambda: float = 0.5,
    finish_smooth_iterations: int = 10,
) -> tuple[trimesh.Trimesh, PostprocessStats]:
    """배경 파편 정리 + 스무딩 단계 전부를 엮는다.

    Args:
        keep_largest(True): 가장 큰 연결 요소만 남기고 부유 파편 제거.
        smooth_boundary_enabled(True): 열린 경계(발목 절단면 등) 테두리를
            `smooth_boundary_loops()`로 다듬어 톱니 모양을 완화한다.
        fill_holes(True): 작은 구멍(핀홀)만 메움 -- 큰 구멍은 그대로.
        fill_round_holes_enabled(False): 원형 구멍(단순 미관측 결손 추정)은
            크더라도 메움(`fill_round_holes()` 참고) -- 발목 절단면처럼
            원래 열려있어야 할 구멍과 헷갈릴 소지가 있어 아직 기본 꺼짐,
            결과를 확인하며 쓸 것.
        sand_surface_enabled(True): 전체 정점 이차곡면 투영 노이즈 완화.
        smooth_high_curvature(True): 고곡률 영역 국소 Taubin 스무딩.
        finish_smooth(True): 라플라시안 마감 스무딩 -- 위 단계로 안 빠지는
            잔여 고주파 표면 노이즈 정리. 비용 미미(정점 10만개 기준 수 초).

    Returns:
        (후처리된 메쉬, 단계별 통계).
    """
    stats = PostprocessStats(faces_before=len(mesh.faces))

    if keep_largest:
        mesh, faces_before, faces_after = keep_largest_component(mesh)
        stats.faces_after_largest_component = faces_after
        if faces_after < faces_before:
            print(f"[finishing] 부유 파편 제거: {faces_before:,} -> {faces_after:,} faces")
            stats.steps_applied.append("keep_largest_component")
    else:
        stats.faces_after_largest_component = len(mesh.faces)

    if fill_holes:
        mesh, n_filled = fill_small_holes(mesh, max_hole_diameter_ratio=fill_holes_max_diameter_ratio)
        stats.n_holes_filled = n_filled
        if n_filled:
            print(f"[finishing] 작은 구멍 메움: {n_filled}개")
            stats.steps_applied.append("fill_small_holes")

    if fill_round_holes_enabled:
        mesh, n_round_filled = fill_round_holes(
            mesh, min_circularity=fill_round_holes_min_circularity,
            max_hole_diameter_ratio=fill_round_holes_max_diameter_ratio,
        )
        stats.n_round_holes_filled = n_round_filled
        if n_round_filled:
            print(f"[finishing] 원형 구멍 메움: {n_round_filled}개")
            stats.steps_applied.append("fill_round_holes")

    if smooth_boundary_enabled:
        mesh = smooth_boundary_loops(
            mesh, iterations=smooth_boundary_iterations, lamb=smooth_boundary_lambda,
        )
        stats.steps_applied.append("smooth_boundary_loops")

    if sand_surface_enabled:
        mesh = sand_surface(
            mesh, min_neighbors=sand_min_neighbors, max_neighbors=sand_max_neighbors,
            iterations=sand_iterations,
        )
        stats.steps_applied.append("sand_surface")

    if smooth_high_curvature:
        mesh = smooth_high_curvature_regions(
            mesh, curvature_percentile=curvature_percentile,
            min_radius_edge_mult=curvature_min_radius_mult, max_radius_edge_mult=curvature_max_radius_mult,
            iterations=curvature_iterations, alpha=curvature_alpha, mu=curvature_mu,
        )
        stats.steps_applied.append("smooth_high_curvature_regions")

    if finish_smooth:
        mesh = finish_smooth_mesh(mesh, lamb=finish_smooth_lambda, iterations=finish_smooth_iterations)
        stats.steps_applied.append("finish_smooth_mesh")

    return mesh, stats
