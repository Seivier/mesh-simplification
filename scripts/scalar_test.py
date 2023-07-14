import openmesh
import polyscope as ps
import numpy as np

from util.simplification import simplification


def main():
    target = "cat10.off"
    original_mesh = openmesh.read_trimesh(f"data/{target}")

    points = np.zeros((original_mesh.n_vertices(), 4))
    for i, p in enumerate(original_mesh.points()):
        x = p[0]
        y = p[1]
        z = p[2]
        u = (x**2 + y**2 + z**2)
        points[i] = np.array([x, y, z, u])


    simp2_mesh, simp_points = simplification(original_mesh, points, target_v=5000, valence_aware=False)

    original_values = np.zeros((original_mesh.n_vertices()))
    for i in range(original_mesh.n_vertices()):
        original_values[i] = points[i][3]
    #
    simp_values = np.zeros((simp2_mesh.n_vertices()))
    for i in range(simp2_mesh.n_vertices()):
        simp_values[i] = simp_points[i][3]

    ps.init()
    mo = ps.register_surface_mesh("original", original_mesh.points(), original_mesh.face_vertex_indices())
    mo.add_scalar_quantity("original", original_values, enabled=True)
    #
    ms = ps.register_surface_mesh("simplificated", simp2_mesh.points(), simp2_mesh.face_vertex_indices(), enabled=False)
    ms.add_scalar_quantity("simplificated", simp_values, enabled=True)
    #
    ps.show()


if __name__ == "__main__":
    main()