import argparse
import os
import openmesh
import polyscope as ps

from util.simplification import simplification


def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, default='data/cat0.off')
    parser.add_argument("--v", type=int, default=1000)
    parser.add_argument("--optim", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    mesh_name = os.path.basename(args.input).split(".")[-2]

    original_mesh = openmesh.read_trimesh(args.input)
    simp2_mesh, simp_points = simplification(original_mesh, original_mesh.points(), target_v=args.v, valence_aware=args.optim)
    openmesh.write_mesh(f"data/{mesh_name}.obj", original_mesh)
    openmesh.write_mesh(f"output/{mesh_name}-simplificated-{args.v}.obj", simp2_mesh)


if __name__ == "__main__":
    main()