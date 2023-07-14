import random
import openmesh
import polyscope as ps
import numpy as np

from util.simplification import simplification


def main():
    target = "horse6.off"
    original_mesh = openmesh.read_trimesh(f"data/{target}")
    # Openmesh calculates the normals of the og mesh
    original_mesh.request_vertex_normals()
    original_mesh.request_face_normals()
    original_mesh.update_normals()
    original_mesh.release_face_normals()

    points = np.zeros((original_mesh.n_vertices(), 6))
    for i, p in enumerate(original_mesh.vertices()):
        points[i] = np.concatenate([original_mesh.point(p), original_mesh.normal(p)])


    simp2_mesh, simp_points = simplification(original_mesh, points, target_v=500, valence_aware=False)

    original_normals = np.zeros((original_mesh.n_vertices(), 3))
    for i in range(original_mesh.n_vertices()):
        original_normals[i] = points[i][3:]
    #
    simp_normals = np.zeros((simp2_mesh.n_vertices(), 3))
    for i in range(simp2_mesh.n_vertices()):
        simp_normals[i] = simp_points[i][3:]

    ps.init()
    mo = ps.register_surface_mesh("original", original_mesh.points(), original_mesh.face_vertex_indices())
    mo.add_vector_quantity("original", original_normals, enabled=True)
    #
    ms = ps.register_surface_mesh("simplificated", simp2_mesh.points(), simp2_mesh.face_vertex_indices(), enabled=False)
    ms.add_vector_quantity("simplificated", simp_normals, enabled=True)
    #
    ps.show()


if __name__ == "__main__":
    main()