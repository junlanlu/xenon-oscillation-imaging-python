# supposed to be a replicate of sparse_gridding_distance.py
import ctypes as ct
import math
import os
import pdb
import platform
import sys

import numpy as np

DEBUG = False
DEBUG_GRID = False


def grid_point(
    sample_loc,
    idx_convert,
    kernel_halfwidth_sqr,
    ndims,
    cur_dim,
    bounds,
    seed_pt,
    kern_dist_sq,
    output_dims,
    sample_index,
    n_nonsparse_entries,
    sparse_sample_indices,
    sparse_voxel_indices,
    sparse_distances,
    force_dim,
):
    lower = int(bounds[2 * cur_dim])
    upper = int(bounds[2 * cur_dim + 1])
    new_kern_dist_sq = 0.0
    # print(lower, upper)
    for i in range(lower, upper + 1):
        seed_pt[cur_dim] = i
        if (cur_dim == force_dim) or force_dim == -1:
            if DEBUG_GRID:
                print(
                    "\t\tSaving distance for dim %u = %f (adding %f to %f)"
                    % (
                        cur_dim + 1,
                        new_kern_dist_sq,
                        new_kern_dist_sq * new_kern_dist_sq,
                        kern_dist_sq,
                    )
                )

            new_kern_dist_sq = float(i - sample_loc[cur_dim])
            new_kern_dist_sq *= new_kern_dist_sq
            new_kern_dist_sq += kern_dist_sq
        else:
            if DEBUG_GRID:
                print(
                    "\t\tWrong dimension dim %u, maintaining distance %f"
                    % (cur_dim + 1, kern_dist_sq)
                )

            new_kern_dist_sq = kern_dist_sq

        if cur_dim > 0:
            if DEBUG_GRID:
                for j in range(ndims):
                    if j < cur_dim:
                        print("%u:%u" % (bounds[2 * j], bounds[2 * j + 1]))
                    else:
                        print("%u" % seed_pt[j])
                    if j < ndims - 1:
                        print(",")

            grid_point(
                sample_loc,
                idx_convert,
                kernel_halfwidth_sqr,
                ndims,
                cur_dim - 1,
                bounds,
                seed_pt,
                new_kern_dist_sq,
                output_dims,
                sample_index,
                n_nonsparse_entries,
                sparse_sample_indices,
                sparse_voxel_indices,
                sparse_distances,
                force_dim,
            )
        else:
            if new_kern_dist_sq <= kernel_halfwidth_sqr:
                idx_ = 0
                for j in range(ndims):
                    idx_ += seed_pt[j] * idx_convert[j]

                # print(sample_index, n_nonsparse_entries)
                sparse_sample_indices[n_nonsparse_entries] = sample_index + 1
                sparse_voxel_indices[n_nonsparse_entries] = float(idx_ + 1)
                sparse_distances[n_nonsparse_entries] = math.sqrt(new_kern_dist_sq)
                # pdb.set_trace()
                n_nonsparse_entries[0] += 1

            else:
                if DEBUG_GRID:
                    print(
                        "\tVoxel [%u, %u, %u] is too far (%f > %f) from sample"
                        + " point [%u, %u, %u] (index %u)!"
                        % (
                            seed_pt[0],
                            seed_pt[1],
                            seed_pt[2],
                            new_kern_dist_sq,
                            kernel_halfwidth_sqr,
                            sample_loc[0],
                            sample_loc[1],
                            sample_loc[2],
                            sample_index,
                        )
                    )
                    # rewrite this
        seed_pt[cur_dim] = lower


def sparse_gridding_distance(
    coords,
    kernel_width,
    npts,
    ndims,
    output_dims,
    n_nonsparse_entries,
    max_size,
    force_dim,
):
    kernel_halfwidth = kernel_width * 0.5
    kernel_halfwidth_sqr = kernel_halfwidth**2
    output_halfwidth = np.zeros(ndims)
    nonsparse_sample_indices = np.zeros(max_size)
    nonsparse_voxel_indices = np.zeros(max_size)
    nonsparse_distances = np.zeros(max_size)

    for dim in range(ndims):
        output_halfwidth[dim] = int(np.ceil(float(output_dims[dim] * 0.5)))

    idx_convert = np.zeros(ndims)
    for dim in range(ndims):
        idx_convert[dim] = 1
        if dim > 0:
            for p in range(dim):
                idx_convert[dim] = idx_convert[dim] * output_dims[p]

    bounds = np.zeros(2 * ndims)
    seed_pt = np.zeros(ndims)
    sample_loc = np.zeros(ndims)

    for p in range(npts):
        if p % 1000 == 0:
            print(p)
        for dim in range(ndims):
            sample_loc[dim] = coords[ndims * p + dim] * float(output_dims[dim]) + float(
                output_halfwidth[dim]
            )
            bounds[2 * dim] = int(max(np.ceil(sample_loc[dim] - kernel_halfwidth), 0))
            bounds[2 * dim + 1] = int(
                min(np.floor(sample_loc[dim] + kernel_halfwidth), output_dims[dim] - 1)
            )
            seed_pt[dim] = bounds[2 * dim]

        if DEBUG:
            print("GRIDDING ")
            print(
                "Sample loc = [%f,%f,%f], "
                % (sample_loc[0], sample_loc[1], sample_loc[2])
            )
            print(
                "Bounds     = [%u:%u,%u:%u,%u:%u]"
                % (bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
            )

        grid_point(
            sample_loc,
            idx_convert,
            kernel_halfwidth_sqr,
            ndims,
            ndims - 1,
            bounds,
            seed_pt,
            0,
            output_dims,
            p,
            n_nonsparse_entries,
            nonsparse_sample_indices,
            nonsparse_voxel_indices,
            nonsparse_distances,
            force_dim,
        )

    # if DEBUG:
    #     for p in range(n_nonsparse_entries[0]):
    #         print("nsp=%u  vox=%f, sample %f, distance=[%f]" % (p,nonsparse_voxel_indices[p],
    #             nonsparse_sample_indices[p], nonsparse_distances[p*ndims]))

    return nonsparse_sample_indices + nonsparse_voxel_indices + nonsparse_distances


def sparse_gridding_c(traj, kernel_para, matrix_size, force_dim):

    # wrap up c function:
    # void sparse_gridding_distance(double *coords, double kernel_width,
    #       unsigned int npts, unsigned int ndims,
    #       unsigned int *output_dims,
    #       unsigned int *n_nonsparse_entries,
    #       unsigned int max_size,
    #       int force_dim)

    lib_path = os.path.join(os.path.dirname(__file__), "..", "bin", "libsparse.so")
    _sparse = ct.CDLL(lib_path)
    _sparse.sparse_gridding_distance.argtypes = (
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.c_uint,
        ct.c_uint,
        ct.POINTER(ct.c_uint),
        ct.POINTER(ct.c_uint),
        ct.c_uint,
        ct.c_int,
    )
    npts, ndim = np.shape(traj)
    # flatten traj to a list for input
    traj = traj.flatten().tolist()
    kernel_para = kernel_para
    matrix_size = matrix_size.astype(int).flatten().tolist()

    num_coord = len(traj)
    num_matrixsize = len(matrix_size)

    # calculate max size of the output indices
    max_nNeighbors = 1
    for dim in range(0, ndim):
        max_nNeighbors = int(max_nNeighbors * (kernel_para + 1))

    max_size = npts * max_nNeighbors

    # create empty output
    nSparsePoints = [0] * 1

    # define argument types
    coord_type = ct.c_double * num_coord
    outputsize_type = ct.c_uint * num_matrixsize
    n_nonsparse_entries_type = ct.c_uint * 1

    # set_result to return an array of numbers. This is deprecated.
    # _sparse.sparse_gridding_distance.restype = ndpointer(
    #     dtype=ct.c_double, shape=(max_size*3,))

    _sparse.sparse_gridding_distance.restype = ct.POINTER(ct.c_double * (max_size * 3))
    result = _sparse.sparse_gridding_distance(
        coord_type(*traj),
        ct.c_double(kernel_para),
        ct.c_uint(npts),
        ct.c_uint(ndim),
        outputsize_type(*matrix_size),
        n_nonsparse_entries_type(*nSparsePoints),
        ct.c_uint(max_size),
        ct.c_int(force_dim),
    )
    # convert pointer to np array
    result = np.asarray(np.ctypeslib.as_array(result.contents, shape=(max_size * 3,)))
    sample_indices = result[:max_size]
    voxel_indices = result[max_size : 2 * max_size]
    distances = result[2 * max_size : 3 * max_size]

    return sample_indices, voxel_indices, distances
