#!/usr/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, signal, glob
import open3d as o3d
import cvxpy as cp

from functions import find_neighbors, loss, check_error_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cl', type=str, default='/home/ncos/Desktop/3d_dvs_datasets/simple/seq_03.npz', required=False)
    parser.add_argument('--z_start', type=float, required=False, default=1.0)
    parser.add_argument('--z_end', type=float, required=False, default=1.5)
    parser.add_argument('--z_scale', type=float, required=False, default=100)

    #parser.add_argument('--num_fixed_points', type=int, required=False, default=20000)
    #parser.add_argument('--num_points', type=int, required=False, default=100)
    parser.add_argument('--num_neighbors', type=float, required=False, default=16)
    parser.add_argument('--window_size', type=float, required=False, default=1000)


    args = parser.parse_args()

    cl = args.cl
    z_start = args.z_start
    z_end = args.z_end
    z_scale = args.z_scale


    # Read the data
    sl_npz         = np.load(cl)
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']

    # Scale up z axis (it is in seconds, not in pixels)
    cloud[:,0] *= z_scale

    # Print stats
    print ("Event (point) count:", cloud.shape[0])
    print ("min/max time (z):", cloud[0][0], "/", cloud[-1][0])

    # Cloud is stored as a point array, 4 components per point:
    # (time, x, y, polarity)
    # we treat time as z axis
    # polarity indicates whether brightness on the pixel (x, y)
    # went up or down; we can ignore that for now

    # To access the z axis efficiently, the lookup table 'idx'
    # is provided. The index of the point with z = 'alpha' is between
    # idx['alpha' // discretization] and idx['alpha' // discretization + 1]

    # Now, figure out slice (z axis range) to process
    if (z_start < 0 or z_end < 0 or z_start > z_end or z_end > cloud[-1][0]):
        print ("Please specify start and end event timestamps (z_start/z_end) within event cloud range")
        sys.exit(0)

    # Watch out, this might be out of bounds
    start_index = idx[int(z_start / discretization)]
    end_index   = idx[int(z_end // discretization)]

    # Chop off the array of interest
    cloud_slice = cloud[start_index : end_index]
    cloud_slice[0, :] -= cloud_slice[0, 0]
    print ("Chopped event (point) count:", cloud_slice.shape[0])

    # Generate a random? feasible SOCP.
    n_fixed = idx[int((z_start + 0.01) / discretization)] - start_index   #args.num_fixed_points
    n_points = idx[int((z_start + 0.025) / discretization)] - start_index - n_fixed  #args.num_points
    k_neigh = args.num_neighbors
    window_size = args.window_size
    cloud_new = cloud_slice[:,[0,1,2]][0:n_fixed+n_points]
    print ("Fixed / moving points:", n_fixed, n_points)

    # Convert to Open3D PCD
    # http://www.open3d.org/docs/release/tutorial/Basic/working_with_numpy.html
    xyz = np.dstack((cloud_slice[:,1], cloud_slice[:,2], cloud_slice[:,0]))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    np_colors = np.array([[1.0,0,0]] * xyz.shape[0])
    np_colors[n_fixed:n_fixed + n_points] = np.array([0.0, 0.0, 1.0])
    np_colors[n_fixed + n_points:] = np.array([0.0, 0.0, 0.0])
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    o3d.visualization.draw_geometries([pcd])

    n_constr = n_points * 3
    p = 1
    N, D = find_neighbors(cloud_new, k_neigh, window_size, n_fixed)

    # linear program with quadratic constraints
    f = -np.array([1,0,0]*n_points)
    A = []
    b = []
    c = []
    d = []
    x0 = np.reshape(cloud_new[n_fixed:], (3*n_points,1))
    for index, point in enumerate(cloud_new[n_fixed:]):
        index_shifted = index+n_fixed
        neighb_ind = N[index_shifted]
        dist = D[index_shifted]
        # for each neighbor point
        for j in range(len(dist)):

            # if the neighbor is a fixed point then constraint is the distance to it
            if neighb_ind[j] < n_fixed:

                A_new = np.zeros((3, 3*n_points))
                A_new[0,3*index] = 1
                A_new[1,3*index+1] = 1
                A_new[2,3*index+2] = 1
                b_new = -cloud_new[neighb_ind[j]]
            else:

                neighb_ind_var = neighb_ind[j] - n_fixed
                A_new_1 = np.zeros((3, 3*n_points))
                A_new_1[0,3*index] = 1
                A_new_1[1,3*index+1] = 1
                A_new_1[2,3*index+2] = 1
                A_new_2 = np.zeros((3, 3*n_points))
                A_new_2[0,3*neighb_ind_var] = 1
                A_new_2[1,3*neighb_ind_var+1] = 1
                A_new_2[2,3*neighb_ind_var+2] = 1
                A_new = A_new_1-A_new_2
                b_new = np.zeros(3)

            A.append(A_new)
            b.append(b_new)
            c.append(np.zeros(3*n_points))
            d.append(dist[j])
    F = np.random.randn(p, 3*n_points)
    g = F@x0


    # Define and solve the CVXPY problem.
    x = cp.Variable(3*n_points)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [cp.SOC(c[i].T@x + d[i], A[i]@x + b[i]) for i in range(k_neigh*n_points)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x), soc_constraints + [F@x == g])
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)

    x_val = np.reshape(x.value, [n_points,3])
    print('error:', check_error_2(cloud_new, x_val, N,D, n_fixed))
    # plot the points
    cloud_optimized = cloud_new
    cloud_optimized[n_fixed:n_fixed+n_points] = x_val
    xyz = np.dstack((cloud_optimized[:,1], cloud_optimized[:,2],
                     cloud_optimized[:,0]))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])
