import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, signal, glob
import open3d as o3d
import torch
from torch.autograd import Variable
from functions import find_neighbors, loss, check_error, check_error_2


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cl', type=str, default='seq_03.npz', required=True)
    parser.add_argument('--z_start', type=float, required=False, default=0.3)
    parser.add_argument('--z_end', type=float, required=False, default=1.5)
    parser.add_argument('--z_scale', type=float, required=False, default=100)
    parser.add_argument('--num_fixed_points', type=int, required=False, default=20000)
    parser.add_argument('--num_points', type=int, required=False, default=1000)
    parser.add_argument('--lam', type=float, required=False, default=1)
    parser.add_argument('--num_neighbors', type=float, required=False, default=3)
    parser.add_argument('--window_size', type=float, required=False, default=10000)
    parser.add_argument('--learning_rate', type=float, required=False, default=1)
    parser.add_argument('--weight_decay', type=float, required=False, default=10**(-4))


    args = parser.parse_args()

    cl = 'seq_03.npz'
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
    print ("Chopped event (point) count:", cloud_slice.shape[0])

    # Convert to Open3D PCD
    # http://www.open3d.org/docs/release/tutorial/Basic/working_with_numpy.html
    xyz = np.dstack((cloud_slice[:,1], cloud_slice[:,2], cloud_slice[:,0]))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])




    # parameters (number of fixed points, number of points to optimize,
    # regularization parameter, number of neighbors and window size)
    n_fixed = args.num_fixed_points
    n_points = args.num_points
    lam = args.lam
    k_neigh = args.num_neighbors
    window_size = args.window_size
    lr = args.learning_rate
    wd = args.weight_decay
    
    
    cloud_new = cloud_slice[0:n_fixed+n_points,[0,1,2]]

    # plot points to optimize
    xyz = np.dstack((cloud_new[:,1], cloud_new[:,2],cloud_new[:,0]))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


    dtype = torch.cuda.FloatTensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = Variable(torch.from_numpy(cloud_new[n_fixed:])).to(device).detach().requires_grad_(True)
    N, D = find_neighbors(cloud_new, k_neigh, window_size, n_fixed)



    optimizer = torch.optim.Adam([x], lr=lr, weight_decay = wd)
    f_x = loss(cloud_new, x, N, D, n_fixed, lam)
    print(x)
    print(f_x)
    for i in range(5000):
        f_x = loss(cloud_new, x, N, D, n_fixed, lam)
        optimizer.zero_grad()
        f_x.backward(retain_graph=True)
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(i + 1, x, f_x)




    x_val = x.cpu().detach().numpy()
    print('error:', check_error(cloud_new, x_val, N,D, n_fixed))
    # plot the points 
    cloud_optimized = cloud_new
    cloud_optimized[n_fixed:n_fixed+n_points] = x_val
    xyz = np.dstack((cloud_optimized[:,1], cloud_optimized[:,2],
                     cloud_optimized[:,0]))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

