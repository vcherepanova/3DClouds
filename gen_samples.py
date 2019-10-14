#!/usr/bin/python3

import argparse
import numpy as np
import scipy
import os, sys, math, signal, glob
import open3d as o3d


def gen_line(x0, y0, x1, y1, z, npts):
    x = x1 - x0
    y = y1 - y0
    l = math.sqrt(x * x + y * y)
    x /= l
    y /= l
    line_arr = []
    for i in range(npts):
        x_curr = x0 + x * i * l / npts
        y_curr = y0 + y * i * l / npts
        line_arr.append([z, x_curr, y_curr])
    return np.array(line_arr, dtype=np.float32)


def getR(theta):
    return np.array(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta),  np.cos(theta))))


def gen_triangle(center, z, scale = (1.0, 1.0), theta = 0.0):
    p0 = np.array([10, - 10], dtype=np.float32)
    p1 = np.array([0, 10], dtype=np.float32)
    p2 = np.array([7, 4], dtype=np.float32)

    p0 *= scale
    p1 *= scale
    p2 *= scale

    R = getR(theta)
    p0 = R.dot(p0) + center
    p1 = R.dot(p1) + center
    p2 = R.dot(p2) + center

    tr = np.vstack((
      gen_line(p0[0], p0[1], p1[0], p1[1], z, 80),
      gen_line(p0[0], p0[1], p2[0], p2[1], z, 80),
      gen_line(p2[0], p2[1], p1[0], p1[1], z, 50)
    ))
    return tr


def gen_circle(center, z, scale = (1.0, 1.0), theta = 0.0):
    circle_arr = []
    base_x = np.array([0, 1], dtype=np.float32)
    for i in np.arange(0.0, 6.28, 0.07):
        R = getR(i)
        xy = R.dot(base_x)
        circle_arr.append([z, xy[0], xy[1]])
    circle = np.array(circle_arr, dtype=np.float32)
    circle[:,1] *= scale[0]
    circle[:,2] *= scale[1]

    for p in circle:
        p += ((0,) + center)

    return circle


def get_sample_shift():
    tr0 = gen_triangle((0, 0), 0, scale=(0.2, 0.2))
    tr1 = gen_triangle((0.02, 0.08), 0.05, scale=(0.2, 0.2))
    return (tr0, tr1)


def get_sample_scale():
    tr0 = gen_triangle((0, 0), 0, scale=(0.2, 0.2))
    tr1 = gen_triangle((-0.05, -0.02), 0.05, scale=(0.21, 0.21))
    return (tr0, tr1)


def get_sample_rot():
    tr0 = gen_triangle((0, 0), 0, scale=(0.2, 0.2), theta=0.0)
    tr1 = gen_triangle((0, 0), 0.05, scale=(0.2, 0.2), theta=0.03)
    return (tr0, tr1)


def get_sample_tear():
    tr0 = gen_triangle((0, 0), 0, scale=(0.2, 0.2), theta=0.0)
    cr0 = gen_circle((1.85, 0), 0, scale=(0.2, 0.2), theta=0.0)
    f0 = np.vstack((tr0, cr0))

    tr1 = gen_triangle((0, 0), 0.05, scale=(0.2, 0.2), theta=0.0)
    cr1 = gen_circle((1.78, 0), 0.05, scale=(0.2, 0.2), theta=0.0)
    f1 = np.vstack((tr1, cr1))

    return (f0, f1)


def visulaize_sample(sample):
    cl = np.vstack(sample)
    xyz = np.dstack((cl[:,1], cl[:,2], cl[:,0]))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    visulaize_sample(get_sample_shift())
    visulaize_sample(get_sample_scale())
    visulaize_sample(get_sample_rot())
    visulaize_sample(get_sample_tear())
