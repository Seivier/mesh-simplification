import openmesh
import numpy as np


def compute_face_centers(mesh: openmesh.TriMesh):
    """
    Computes the centers per face

    :param mesh:
    :return: array with points
    """

    points = mesh.points()
    face_centers = np.zeros((mesh.n_faces(), 3))

    for face in mesh.faces():
        v_it = openmesh.FaceVertexIter(mesh, face)

        v0 = next(v_it).idx()
        v1 = next(v_it).idx()
        v2 = next(v_it).idx()

        face_centers[face.idx()] = np.sum([
            points[v0],
            points[v1],
            points[v2]], 0) / 3

    return face_centers


def compute_face_base(mesh: openmesh.TriMesh, points: np.ndarray[float]):
    """
    Computes the orthogonal vectors per face

    :param mesh:
    :param points:
    :return: array with orthogonal vectors
    """
    dim = points.shape[1]

    vectors = np.zeros((mesh.n_faces(), 3, dim))

    for face in mesh.faces():
        v_it = openmesh.FaceVertexIter(mesh, face)

        v0 = next(v_it).idx()
        v1 = next(v_it).idx()
        v2 = next(v_it).idx()

        p = points[v0]
        q = points[v1]
        r = points[v2]

        e1 = q - p
        e1 /= np.linalg.norm(e1)
        e2 = r - p - (np.dot(e1, r - p) * e1)
        e2 /= np.linalg.norm(e2)

        vectors[face.idx(), 0] = p
        vectors[face.idx(), 1] = e1
        vectors[face.idx(), 2] = e2

    return vectors


def compute_face_normals(mesh: openmesh.TriMesh):
    """
    Computes the normals per face

    :param mesh:
    :return: array with normals
    """

    points = mesh.points()
    normal_faces = np.zeros((mesh.n_faces(), 3))

    for face in mesh.faces():
        v_it = openmesh.FaceVertexIter(mesh, face)

        v0 = next(v_it).idx()
        v1 = next(v_it).idx()
        v2 = next(v_it).idx()

        normal = np.cross(points[v1] - points[v0],
                          points[v2] - points[v0])
        normal /= np.linalg.norm(normal)
        normal_faces[face.idx()] = normal

    return normal_faces


def calculate_vertex_normals(mesh: openmesh.TriMesh):
    """
    Calculates the normals per vertex via adding the face normals.

    :param mesh:
    :return: array with normals
    """
    normal_faces = compute_face_normals(mesh)

    normal_vertices = np.zeros((mesh.n_vertices(), 3))
    for vertex in mesh.vertices():
        face_it = openmesh.VertexFaceIter(mesh, vertex)
        for face in face_it:
            normal_vertices[vertex.idx(), :] += normal_faces[face.idx(), :]

        normal_vertices[vertex.idx(), :] /= np.linalg.norm(normal_vertices[vertex.idx(), :])

    return normal_vertices