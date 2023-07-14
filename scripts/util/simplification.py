
import heapq

import openmesh
import numpy as np
import copy

from util.normals import compute_face_normals, compute_face_centers, compute_face_base


class Quadric:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: float):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, v: np.ndarray):
        qt = np.matmul(np.matmul(v.T, self.a), v)
        lt = 2 * np.matmul(self.b.T, v)
        return qt + lt + self.c

    def __add__(self, other):
        return Quadric(self.a + other.a, self.b + other.b, self.c + other.c)


def get_quadric(p: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    n = len(p)
    te1 = np.reshape(e1, (n, 1))
    te2 = np.reshape(e2, (n, 1))
    a = np.identity(n) - (te1 * te1.T) - (te2 * te2.T)
    b = (p.dot(e1)) * e1 + (p.dot(e2)) * e2 - p
    c = p.dot(p) - np.power(p.dot(e1), 2) - np.power(p.dot(e2), 2)
    return Quadric(a, b, c)


def edge_collapse(simp_mesh: openmesh.TriMesh, simp_points: np.ndarray, simp_mesh_vf, simp_mesh_v2v, vi_0, vi_1, merged_faces, vi_mask, fi_mask,
                  Q_s, E_heap, valence_aware):

    simp_mesh.points()[vi_0] = 0.5 * (simp_mesh.points()[vi_0] + simp_mesh.points()[vi_1])
    simp_points[vi_0] = 0.5 * (simp_points[vi_0] + simp_points[vi_1])
    simp_mesh.points()[vi_1] = np.copy(simp_mesh.points()[vi_0])
    simp_points[vi_1] = np.copy(simp_points[vi_0])
    simp_mesh.collapse(
        simp_mesh.find_halfedge(simp_mesh.vertex_handle(vi_1), simp_mesh.vertex_handle(vi_0))
    )

    # get shared vertices between vi_0 and vi_1, there should be 2
    shared_vv = list(set(simp_mesh_v2v[vi_0]).intersection(set(simp_mesh_v2v[vi_1])))

    # update faces:
    # - merge faces of vi_0 and vi_1 into vi_0, and remove all faces from vi_1
    simp_mesh_vf[vi_0] = simp_mesh_vf[vi_0].union(simp_mesh_vf[vi_1]).difference(merged_faces)
    simp_mesh_vf[vi_1] = set()
    # - remove the old merged faces from the shared_vv, the vertices that had connections with it
    simp_mesh_vf[shared_vv[0]] = simp_mesh_vf[shared_vv[0]].difference(merged_faces)
    simp_mesh_vf[shared_vv[1]] = simp_mesh_vf[shared_vv[1]].difference(merged_faces)
    # - mark merged faces as removed
    fi_mask[np.array(list(merged_faces)).astype(int)] = False

    # update vertices:
    new_vi_0 = set(simp_mesh_v2v[vi_0]).union(set(simp_mesh_v2v[vi_1])).difference({vi_0, vi_1})
    simp_mesh_v2v[vi_0] = list(new_vi_0)
    for v in simp_mesh_v2v[vi_1]:  # disconnect vi_1 neighbors form vi_1 and connnect them to vi_0
        if v != vi_0:
            simp_mesh_v2v[v] = list(set(simp_mesh_v2v[v]).difference({vi_1}).union({vi_1}))
    simp_mesh_v2v[vi_1] = []  # disconnect vi_1
    # - mark vi_1 as removed, while updating vi_0's position as the new merged vertex
    vi_mask[vi_1] = False
    np.delete(simp_points, vi_1, axis=0)

    Q_0 = Q_s[vi_0]
    for vv_i in simp_mesh_v2v[vi_0]:
        # v_mid = 0.5 * (simp_mesh.points()[vi_0] + simp_mesh.points()[vv_i])
        # v_mid = 0.5 * (simp_points[vi_0] + simp_points[vv_i])
        Q_1 = Q_s[vv_i]
        Q_new = Q_0 + Q_1

        try:
            ainv = np.linalg.inv(Q_new.a)
            v_new = -np.matmul(ainv, Q_new.b)
        except:
            v_new = 0.5 * (vi_0 + vv_i)
        # v4_mid = np.concatenate([v_mid, np.array([1])])

        E_new = Q_new(v_new)
        heapq.heappush(E_heap, (E_new, (vi_0, vv_i)))


def simplification(mesh: openmesh.TriMesh, points: np.ndarray[float], target_v: int, valence_aware: bool = True, midpoint: bool = False):
    vs = points
    dim = len(vs[0])
    vf = []
    vf_raw = np.array([item for item in mesh.vertex_face_indices()])
    for row in vf_raw:
        vf.append(set(row[row >= 0]))

    fo = compute_face_base(mesh, points)
    
    edges = []
    for edge in mesh.edges():
        # La idea es que sean tipo [v1_idx, v2_idx]
        he = mesh.halfedge_handle(edge, 0)
        edges.append(np.array([mesh.from_vertex_handle(he).idx(),
                               mesh.to_vertex_handle(he).idx()]))
    edges = np.array(edges)
    print("1. Compute Q for each vertex")
    q_null = Quadric(np.zeros((dim, dim)), np.zeros(dim), 0)
    Q_s: list[Quadric] = [q_null for _ in range(len(vs))]
    E_s = [0 for _ in range(len(vs))]
    for i, v in enumerate(vs):
        f_s = np.array(list(vf[i]))
        quadrics = []

        for f in f_s:
            fo_s = fo[f]  # e1 y e2
            quadrics.append(get_quadric(fo_s[0], fo_s[1], fo_s[2]))

        Q_s[i] = sum(quadrics, q_null)
        # v4 = np.concatenate([v, np.array([1])])  # tiene que ser 4-D
        E_s[i] = Q_s[i](v)

    print("2. Compute E for every possible pair and create heapq")

    E_heap = []
    # completar
    for i, e in enumerate(edges):
        v_0, v_1 = vs[e[0]], vs[e[1]]
        Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
        Q_new = Q_0 + Q_1

        if midpoint:
            v_new = 0.5 * (v_0 + v_1)
        else:
            # Q_lp = np.eye(4)
            # Q_lp[:3] = Q_new[:3]
            # try:
            #     Q_lp_inv = np.linalg.inv(Q_lp)
            #     v4_new = np.matmul(Q_lp_inv, np.array([[0, 0, 0, 1]]).reshape(-1, 1)).reshape(-1)
            try:
                ainv = np.linalg.inv(Q_new.a)
                v_new = -np.matmul(ainv, Q_new.b)

            except:
                v_new = 0.5 * (v_0 + v_1)
                # v4_new = np.concatenate([v_new, np.array([1])])

        E_new = Q_new(v_new)
        heapq.heappush(E_heap, (E_new, (e[0], e[1])))

    print("3. Collapse Minimum-error Vertex")
    simp_mesh = copy.deepcopy(mesh)
    simp_points = copy.deepcopy(vs)
    simp_mesh_vf = copy.deepcopy(vf)
    simp_mesh_v2v = []
    for vh in simp_mesh.vertices():
        simp_mesh_v2v.append(sorted([neighbor.idx() for neighbor in simp_mesh.vv(vh)]))

    vi_mask = np.ones([len(simp_mesh.vertices())]).astype(bool)
    fi_mask = np.ones([len(simp_mesh.faces())]).astype(bool)

    max_error = 0
    while np.sum(vi_mask) > target_v:
        if len(E_heap) == 0:
            print("[WARNING]: Edge cannot be collapsed anymore!")
            break

        E_0, (vi_0, vi_1) = heapq.heappop(E_heap)
        max_error = max(max_error, E_0)

        if not (vi_mask[vi_0] and vi_mask[vi_1]):
            continue

        shared_vv = list(set(simp_mesh_v2v[vi_0]).intersection(set(simp_mesh_v2v[vi_1])))
        merged_faces = simp_mesh_vf[vi_0].intersection(simp_mesh_vf[vi_1])

        if len(shared_vv) != 2:
            """ non-manifold! """
            continue
        elif len(merged_faces) != 2:
            """ boundary """
            continue
        else:
            edge_collapse(simp_mesh, simp_points, simp_mesh_vf, simp_mesh_v2v, vi_0, vi_1, merged_faces, vi_mask, fi_mask,
                          Q_s, E_heap, valence_aware=valence_aware)

    simp_mesh.garbage_collection()  # so openmesh removes all discarded vertices and faces

    print("4. Organize new extra data according to the mesh")
    helper_dic = {}
    for p in simp_points:
        helper_dic[tuple(p[:3])] = p[3:]

    for i, p in enumerate(simp_mesh.points()):
        simp_points[i] = np.concatenate([p, helper_dic[tuple(p)]])

    print(f"Finish, max error: {max_error}.")

    return simp_mesh, simp_points
