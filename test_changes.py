import time
from timeit import repeat
from typing import Optional, Tuple

import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import get_mesh_laplacian, add_self_loops, scatter, to_undirected
from scipy.spatial import Delaunay


def get_mesh_laplacian_backup(
    pos: torch.Tensor,
    face: torch.Tensor,
    normalization: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backup of the original get_mesh_laplacian function for comparison.
    """
    assert pos.size(1) == 3 and face.size(0) == 3

    num_nodes = pos.shape[0]

    def get_cots(left: torch.Tensor, centre: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = torch.einsum('ij, ij -> i', left_vec, right_vec)
        cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
        cot = dot / cross  # cot = cos / sin
        return cot / 2.0  # by definition

    # For each triangle face, get all three cotangents:
    cot_021 = get_cots(face[0], face[2], face[1])
    cot_102 = get_cots(face[1], face[0], face[2])
    cot_012 = get_cots(face[0], face[1], face[2])
    cot_weight = torch.cat([cot_021, cot_102, cot_012])

    # Face to edge:
    cot_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    cot_index, cot_weight = to_undirected(cot_index, cot_weight)

    # Compute the diagonal part:
    cot_deg = scatter(cot_weight, cot_index[0], 0, num_nodes, reduce='sum')
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = torch.cat([cot_weight, -cot_deg], dim=0)

    if normalization is not None:

        def get_areas(left: torch.Tensor, centre: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            central_pos = pos[centre]
            left_vec = pos[left] - central_pos
            right_vec = pos[right] - central_pos
            cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
            area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
            return area / 2.0  # since each corresponding area is counted twice

        # Like before, but here we only need the diagonal (the mass matrix):
        area_021 = get_areas(face[0], face[2], face[1])
        area_102 = get_areas(face[1], face[0], face[2])
        area_012 = get_areas(face[0], face[1], face[2])
        area_weight = torch.cat([area_021, area_102, area_012])
        area_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        area_index, area_weight = to_undirected(area_index, area_weight)
        area_deg = scatter(area_weight, area_index[0], 0, num_nodes, 'sum')

        if normalization == 'sym':
            area_deg_inv_sqrt = area_deg.pow_(-0.5)
            area_deg_inv_sqrt[area_deg_inv_sqrt == float('inf')] = 0.0
            edge_weight = (area_deg_inv_sqrt[edge_index[0]] * edge_weight *
                           area_deg_inv_sqrt[edge_index[1]])
        elif normalization == 'rw':
            area_deg_inv = 1.0 / area_deg
            area_deg_inv[area_deg_inv == float('inf')] = 0.0
            edge_weight = area_deg_inv[edge_index[0]] * edge_weight

    return edge_index, edge_weight

# Setup
torch.manual_seed(42)
N = 100000
points = torch.rand(N, 2)
N_LOOPS = int(1000000 / N)

tri = Delaunay(points)

torch.set_default_device('cuda')
pos = torch.concat([torch.tensor(tri.points), torch.rand((tri.points.shape[0], 1))], dim=1)
face = torch.tensor(tri.simplices, dtype=torch.int64).T

# Time all variants of the mesh laplacian
for normalization in [None, 'sym', 'rw']:
    def benchmark_old():
        return get_mesh_laplacian_backup(pos=pos, face=face, normalization=normalization)

    def benchmark_new():
        return get_mesh_laplacian(pos=pos, face=face, normalization=normalization)

    print(f"Normalization: {normalization}")
    laplacian = get_mesh_laplacian(pos=pos, face=face, normalization=normalization)
    laplacian_old = get_mesh_laplacian_backup(pos=pos, face=face, normalization=normalization)
    print(f"Norm of old minus new indices: {torch.norm(1.0 * laplacian[0] - 1.0 * laplacian_old[0])}")
    print(f"Norm of old minus new values: {torch.norm(laplacian[1] - laplacian_old[1])}")

    time = repeat(benchmark_old, number=N_LOOPS)
    time_per_loop_old = min(time) / N_LOOPS
    print(f"OLD: {N_LOOPS} loops; best of 5: {time_per_loop_old:.4e} sec per loop")

    time = repeat(benchmark_new, number=N_LOOPS)
    time_per_loop_new = min(time) / N_LOOPS
    print(f"NEW: {N_LOOPS} loops; best of 5: {time_per_loop_new:.4e} sec per loop")

    print(f"Speedup: {time_per_loop_old / time_per_loop_new:.2f}x")


# Visualize mesh
plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices)
plt.show()
