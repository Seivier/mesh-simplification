import random
import openmesh
import polyscope as ps
import numpy as np

from util.simplification import simplification


def main():
    target = "horse0.off"
    original_mesh = openmesh.read_trimesh(f"data/{target}")
    points = np.zeros((original_mesh.n_vertices(), 6))
    max_x, max_y, max_z = -1e9, -1e9, -1e9
    min_x, min_y, min_z = 1e9, 1e9, 1e9
    for i, p in enumerate(points):
        coord = original_mesh.points()[i]
        x = coord[1]
        y = coord[0]
        z = coord[0]**2 + coord[1]**2
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)

        points[i] = np.concatenate([coord, np.array([x, y, z])])

    for i, p in enumerate(points):
        points[i][3] = (points[i][3] - min_x) / (max_x - min_x)
        points[i][4] = (points[i][4] - min_y) / (max_y - min_y)
        points[i][5] = (points[i][5] - min_z) / (max_z - min_z)

    simp2_mesh, simp_points = simplification(original_mesh, points, target_v=1000, valence_aware=False)

    original_colors = np.zeros((original_mesh.n_vertices(), 3))
    for i in range(original_mesh.n_vertices()):
        original_colors[i] = points[i][3:]

    simp_color = np.zeros((simp2_mesh.n_vertices(), 3))
    for i in range(simp2_mesh.n_vertices()):
        simp_color[i] = simp_points[i][3:]
    # dic_colors = {}
    # for p in simp_points:
    #     dic_colors[tuple(p[:3])] = p[3:]
    #
    # scolors = np.zeros((simp2_mesh.n_vertices(), 3))
    # for i, p in enumerate(simp2_mesh.points()):
    #     scolors[i] = dic_colors[tuple(p)]

    ps.init()
    mo = ps.register_surface_mesh("original", original_mesh.points(), original_mesh.face_vertex_indices())
    mo.add_color_quantity("original", original_colors, enabled=True)

    ms = ps.register_surface_mesh("simplificated", simp2_mesh.points(), simp2_mesh.face_vertex_indices(), enabled=False)
    ms.add_color_quantity("simplificated", simp_color, enabled=True)

    ps.show()


if __name__ == "__main__":
    main()