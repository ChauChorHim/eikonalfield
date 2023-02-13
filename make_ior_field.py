import pickle
import os
import numpy as np

from scipy.ndimage import convolve

from run_nerf import config_parser


def main():
    parser = config_parser()
    args = parser.parse_args()

    voxel_grid_objects = []
    with (open(os.path.join(args.datadir, "voxel_grid.pkl"), "rb")) as openfile:
        while True:
            try:
                voxel_grid_objects.append(pickle.load(openfile))
            except EOFError:
                break

    # Compute gradient of IoR field
    voxel_grid = voxel_grid_objects[0]['data']

    k_s = 4
    ior_field = voxel_grid
    ior_field[ior_field == 1.0] = 1.505
    ior_field[ior_field == 0.0] = 1.0

    kernel = np.ones(k_s*k_s*k_s).reshape(k_s, k_s, k_s)
    ior_field = convolve(ior_field, kernel) / 64.0

    grad_ior_field = np.gradient(ior_field)

    # Compute voxel grid in real world scale
    max_point = voxel_grid_objects[0]['max_point']
    min_point = voxel_grid_objects[0]['min_point']
    grid_size = voxel_grid_objects[0]['num_voxels']
    X, Y, Z = np.meshgrid(np.linspace(0, 1, grid_size),
                          np.linspace(0, 1, grid_size),
                          np.linspace(0, 1, grid_size))
    x_max, y_max, z_max = max_point
    x_min, y_min, z_min = min_point
    X = X * (x_max - x_min) + x_min
    Y = Y * (y_max - y_min) + y_min
    Z = Z * (z_max - z_min) + z_min
    voxel_grid_real_scale = np.concatenate(np.stack([X.reshape(-1, 1),
                                                     Y.reshape(-1, 1),
                                                     Z.reshape(-1, 1)], axis=-1))

    with open(os.path.join(args.datadir, 'grad_ior_field.pkl'), 'wb') as f:
        pickle.dump({
            "grad_ior_field": grad_ior_field,
            "voxel_grid_real_scale": voxel_grid_real_scale,
            "extent": 0,
            "min_point": voxel_grid_objects[0]['min_point'],
            "max_point": voxel_grid_objects[0]['max_point'],
            "num_voxels": voxel_grid_objects[0]['num_voxels'],
        }, f)

    # pass

    # grid_size = voxel_grid_objects[0]['num_voxels']
    # x_max = y_max = z_max = 0.4
    # x_min = y_min = z_min = -0.4
    #
    # X, Y, Z = np.meshgrid(np.linspace(0, 1, grid_size),
    #                       np.linspace(0, 1, grid_size),
    #                       np.linspace(0, 1, grid_size))
    #
    # X = X * (x_max - x_min) + x_min
    # Y = Y * (y_max - y_min) + y_min
    # Z = Z * (z_max - z_min) + z_min
    #
    # import pandas as pd
    # from pyntcloud import PyntCloud
    #
    # mask = np.where(ior_field > 1.0)
    # test = ior_field[mask]
    #
    # cloud = PyntCloud(pd.DataFrame(
    #     # same arguments that you are passing to visualize_pcl
    #     data=np.hstack((X[mask].reshape(-1, 1),
    #                     Y[mask].reshape(-1, 1),
    #                     Z[mask].reshape(-1, 1),
    #                     ior_field[mask].reshape(-1, 1) - ior_field[mask].reshape(-1, 1),
    #                     ior_field[mask].reshape(-1, 1) - ior_field[mask].reshape(-1, 1),
    #                     ior_field[mask].reshape(-1, 1)-1)),
    #     columns=["x", "y", "z", "red", "green", "blue"]))
    #
    # cloud.to_file(os.path.join(args.datadir, f'ior_{grid_size}.ply'))

if __name__ == '__main__':
    main()
