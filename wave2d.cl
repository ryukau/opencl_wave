uint get_index_nd() {
    const int dimension = get_work_dim();
    int index = get_global_id(0);
    int size = 1;
    for (int dim = 1; dim < dimension; ++dim) {
        size *= get_global_size(dim - 1);
        index += size * get_global_id(dim);
    }
    return index;
}

__kernel void fill_boundary(__global uchar *boundary) {
    const int dimension = get_work_dim();
    for (int dim = 0; dim < dimension; ++dim) {
        int gid = get_global_id(dim);
        int size = get_global_size(dim);
        if (gid == 0 || gid == size - 1) {
            boundary[get_index_nd()] = 1;
            return;
        }
    }
}

__kernel void fill_neighbor_2d(__global uint4 *neighbor) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int index = x + len_x * y;

    neighbor[index].s0 = index - 1;
    neighbor[index].s1 = index + 1;
    neighbor[index].s2 = index - len_x;
    neighbor[index].s3 = index + len_x;
}

__kernel void wave2d_jacobi(
    __global uchar *boundary,
    __global uint4 *neighbor,
    __global float *wave0,
    __global float *wave1,
    __global float *wave2,
    const float C0,
    const float C1,
    const float attenuation,
    const int num_jacobi_iter,
    const int step
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int len_y = get_global_size(1);
    const int index = x + len_x * y;

    if (boundary[index] == 1) return;

    // Pick surface.
    if (step < 64 && x == len_x / 2 && y == len_y / 2)
        wave0[index] = 1.0;

    // Roll.
    wave2[index] = wave1[index];
    wave1[index] = attenuation * wave0[index];

    // Jacobi method.
    const float b = (2 * wave1[index] - wave2[index]) / C1;

    const int i_left = neighbor[index].s0;
    const int i_right = neighbor[index].s1;
    const int i_top = neighbor[index].s2;
    const int i_bottom = neighbor[index].s3;

    for (int i = 0; i < num_jacobi_iter; ++i) {
        wave0[index] = b - C0 * (
          wave0[i_left]
          + wave0[i_right]
          + wave0[i_top]
          + wave0[i_bottom]
        );
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
