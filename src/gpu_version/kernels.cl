

// ~~~~~~~~~~~~ KERNEL FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// cubic spline kernel function
float w(float2 x_i, float2 x_j, float alpha, float h) {
  float q = distance(x_i, x_j) / h;
  float t1 = max(0.0, 1.0 - q);
  float t2 = max(0.0, 2.0 - q);
  return alpha * (t2 * t2 * t2 - 4.0 * t1 * t1 * t1);
}

float2 dw(float2 x_i, float2 x_j, float alpha, float h) {
  float q = distance(x_i, x_j) / h;
  float t1 = max(0.0, 1.0 - q);
  float t2 = max(0.0, 2.0 - q);
  float magnitude = (alpha / h) * (-3.0 * t2 * t2 + 12.0 * t1 * t1);
  return (x_i - x_j) * magnitude;
}

// ~~~~~~~~~~~~ SPH APPROXIMATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__kernel void eulercromerstep(
    // arrays
    __global float2 *pos, __global float2 *vel, __global float2 *acc,
    // atomics/variables
    __global float *dt, uint n) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  // float2 new_v = vel[i] + acc[i]*dt;
  vel[i] += acc[i] * dt[0];
  pos[i] += vel[i] * dt[0];
}

__kernel void enforce_boundary(
    // arrays
    __global float2 *pos, __global float2 *vel, __global float2 *acc,
    // constants
    float2 min, float2 max, uint n) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  if (pos[i].x <= min.x) {
    pos[i].x = min.x;
    acc[i].x = 0.0;
    vel[i].x = 0.0;
  }
  if (pos[i].y <= min.y) {
    pos[i].y = min.y;
    acc[i].y = 0.0;
    vel[i].y = 0.0;
  }
  if (pos[i].x >= max.x) {
    pos[i].x = max.x;
    acc[i].x = 0.0;
    vel[i].x = 0.0;
  }
  if (pos[i].y >= max.y) {
    pos[i].y = max.y;
    acc[i].y = 0.0;
    vel[i].y = 0.0;
  }
}

/// n is the number of particles
/// cells and indices are the two components of the handles-array
/// neighbours is a flat array of chunks of 3, containing indices
///  into the handles array
__kernel void update_densities_pressures(
    // atomics
    float k, float rho_0,
    // arrays
    __global float *den, __global float *prs, __global float2 *pos,
    __global uint2 *handles, __global int *neighbours, __global float2 *bdy,
    __global uint2 *bdy_handles, __global int *bdy_neighbours,
    // constants
    float mass, float alpha, float h, uint n, uint n_bdy) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  float new_den = 0.0;
  float2 p = pos[i];

  // sum over fluid neighbours
  for (int k = 0; k < 3; k++) {
    int j = neighbours[3 * i + k]; // j is an index into `handles`
    if (j >= 0) {
      int initial_cell = (handles[j][0]);
      while (j < n && (handles[j][0]) < initial_cell + 3) {
        new_den += w(p, pos[handles[j][1]], alpha, h);
        j++;
      }
    }
  }
  // sum over boundary neighbours
  for (int k = 0; k < 3; k++) {
    int j = bdy_neighbours[3 * i + k]; // j is an index into `bdy_handles`
    if (j >= 0) {
      int initial_cell = (bdy_handles[j][0]);
      while (j < n_bdy && (bdy_handles[j][0]) < initial_cell + 3) {
        new_den += w(p, bdy[bdy_handles[j][1]], alpha, h);
        j++;
      }
    }
  }
  new_den *= mass;
  den[i] = new_den;
  prs[i] = max(0.0, k * (new_den / rho_0 - 1.));
  // prs[i] = k*(new_den/rho_0-1.);
}

__kernel void add_pressure_acceleration(
    // atomics
    float rho_0,
    // arrays
    __global float2 *pos, __global float2 *acc, __global float *den,
    __global float *prs, __global uint2 *handles, __global int *neighbours,
    __global float2 *bdy, __global uint2 *bdy_handles,
    __global int *bdy_neighbours,
    // constants
    float mass, float alpha, float h, uint n, uint n_bdy) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  float2 x_i = pos[i];
  float2 force = (float2)(0.0f, 0.0f);
  float2 force_bdy = (float2)(0.0f, 0.0f);
  float p_i_over_rho_i_squared = prs[i] / (den[i] * den[i]);

  // sum over fluid neighbours
  for (int k = 0; k < 3; k++) {
    int j = neighbours[3 * i + k]; // j is an index into `handles`
    if (j >= 0) {
      int initial_cell = handles[j][0];
      while (j < n && (handles[j][0]) < initial_cell + 3) {
        uint neighbour = handles[j][1];
        // symmetric formula for pressure forces
        force -= dw(x_i, pos[neighbour], alpha, h) *
                 (p_i_over_rho_i_squared +
                  prs[neighbour] / (den[neighbour] * den[neighbour]));
        j++;
      }
    }
  }
  // sum over boundary neighbours
  for (int k = 0; k < 3; k++) {
    int j = bdy_neighbours[3 * i + k]; // j is an index into `handles`
    if (j >= 0) {
      int initial_cell = bdy_handles[j][0];
      while (j < n_bdy && (bdy_handles[j][0]) < initial_cell + 3) {
        uint neighbour = bdy_handles[j][1];
        // symmetric formula for pressure forces
        force_bdy += dw(x_i, bdy[neighbour], alpha, h);
        j++;
      }
    }
  }
  force_bdy *= -2.0f * (prs[i] / (rho_0 * rho_0));
  acc[i] += (force + force_bdy) * mass;
}

__kernel void apply_gravity_viscosity(
    // atomics
    float nu, float g,
    // arrays
    __global float2 *acc, __global float2 *vel, __global float2 *pos,
    __global float *den, __global uint2 *handles, __global int *neighbours,
    // constants
    float mass, float alpha, float h, uint n) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  float2 acc_g = (float2)(0.0, g);
  float2 vis = (float2)(0.0, 0.0);
  float2 x_i = pos[i];
  float2 v_i = vel[i];

  // sum over neighbours
  for (int k = 0; k < 3; k++) {
    int j = neighbours[3 * i + k]; // j is an index into `handles`
    if (j >= 0) {
      int initial_cell = handles[j][0];
      while (j < n && (handles[j][0]) < initial_cell + 3) {
        float2 x_i_j = x_i - pos[handles[j][1]];
        float2 v_i_j = v_i - vel[handles[j][1]];
        float squared_dist = length(x_i_j) * length(x_i_j);
        vis += (mass / den[handles[j][1]]) *
               (dot(v_i_j, x_i_j) / (squared_dist + 0.01f * h * h)) *
               dw(x_i, pos[handles[j][1]], alpha, h);
        j++;
      }
    }
  }
  vis *= nu * 2.0f;
  acc[i] = acc_g + vis;
}

// ~~~~~~~~~~~~ NEIGHBOURHOOD SEARCH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline uint cell_key(float2 p, float ks, float2 min) {
  uint2 key = convert_uint2_rtn(convert_ushort2_rtn((p - min) / ks));
  return (key.y << 16) + key.x;
}

// unit-test the equivalence of the CPU and GPU versions of
// cell-key using the following kernel
__kernel void test_cell_key(__global float2 *pos, __global uint *key, uint n,
                            float ks, float2 min) {
  uint i = get_global_id(0);
  if (i < n) {
    key[i] = cell_key(pos[i], ks, min);
  }
}

__kernel void compute_cell_keys(
    // atomics
    __global float2 *pos_min, float ks,
    // arrays
    __global float2 *pos, __global uint2 *handles, uint n) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  handles[i].x = cell_key(pos[handles[i].y], ks, pos_min[0]);
}

inline int binary_search(__global uint2 *handles, uint key, uint n) {
  // binary search for key
  int low = 0;
  int high = n - 1;
  while (low <= high) {
    int mid = (high - low) / 2 + low;
    uint hit = handles[mid].x;
    if ((key <= hit && hit < key + 3) &&
        (mid == 0 ||
         !(key <= handles[mid - 1].x && handles[mid - 1].x < key + 3))) {
      return mid;
    } else {
      if (hit < key) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
  }
  return -1;
}

__kernel void compute_neighbours(__global const float2 *pos_min,
                                 // arrays
                                 __global const float2 *pos,
                                 __global const uint2 *handles,
                                 __global int *neighbours,
                                 // constants
                                 float const ks, uint const n) {
  int i = get_global_id(0);
  if (i >= n) {
    return;
  }
  const float2 rows_nearby[3] = {
      (float2)(-ks, -ks),
      (float2)(-ks, 0),
      (float2)(-ks, ks),
  };
  for (int k = 0; k < 3; k++) {
    // update neighbour set
    uint key = cell_key(pos[i] + rows_nearby[k], ks, pos_min[0]);
    neighbours[i * 3 + k] = binary_search(handles, key, n);
  }
}

// ~~~~~~~~~~~~ RADIX SORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_histograms(
    __global uint2 *input,      // must be of size n
    __global uint2 *output,     // must be of size n
    __global uint *global_hist, // must be 256 entries per workgroup
    __global uint *counts,      // must be 256 entries
    __local uint *local_hist,   // must be of size 256 for each workgroup
    uint shift,                 // must be one of [0,8,16,24]
    uint n) {
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n_groups = get_num_groups(0);

  // initialize local histogram to zeros
  local_hist[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  // calculate key using selected byte and enter it
  // in the local histogram
  if (gid < n) {
    uint key = ((input[gid].x) >> shift) & 255;
    atomic_inc(&local_hist[key]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // write back histograms to global memory to compute the global prefix sum
  global_hist[lid * n_groups + wid] = local_hist[lid];
  atomic_add(&counts[lid], local_hist[lid]);
}

// USE FOR ARRAY SIZES LESS THAN 2^18 = 262_144
__kernel void radix_sort_prefixsum_small(
    __global uint *global_hist, // must be 256 entries per workgroup
    __local uint *local_hist,   // must be n_groups large
    __global uint *counts,      // must be 256 entries
    uint n_groups) {
  // compute exclusive prefix sum of all histograms
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  uint gid = get_global_id(0);
  if (gid == 0) {
    uint prefix_sum = 0;
    for (uint i = 0; i < 256; i++) {
      uint temp = counts[i];
      counts[i] = prefix_sum;
      prefix_sum += temp;
    }
  }

  // load histogram into shared memory
  if (lid < n_groups) {
    local_hist[lid] = global_hist[lid + wid * n_groups];
  } else {
    local_hist[lid] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // upsweep
  uint stride = 1;
  while (stride < n) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int index = (lid + 1) * stride * 2 - 1;
    if (index < n) {
      local_hist[index] += local_hist[index - stride];
    }
    stride *= 2;
  }

  if (lid == 0) {
    local_hist[n - 1] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // downsweep
  stride = n / 2;
  while (stride >= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int index = (lid + 1) * stride * 2 - 1;
    if (index < n) {
      uint left = local_hist[index - stride];
      local_hist[index - stride] = local_hist[index];
      local_hist[index] += left;
    }
    stride /= 2;
  }

  // write back shared memory to global
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < n_groups) {
    global_hist[lid + wid * n_groups] = local_hist[lid];
  }
}

// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum_a(
    __global uint *global_hist, // must be 256 entries per workgroup
    __local uint *local_hist,   // must be n_groups large
    __global uint *counts,      // must be 256 entries
    __global uint *counts_b,    // must be splinters*256 many
    uint n_groups, uint splinters) {
  // compute exclusive prefix sum of all histograms
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  uint gid = get_global_id(0);
  if (gid == 0) {
    uint prefix_sum = 0;
    for (uint i = 0; i < 256; i++) {
      uint temp = counts[i];
      counts[i] = prefix_sum;
      prefix_sum += temp;
    }
  }

  // compute effective id
  uint padded = splinters * n;
  uint eid = (lid + wid * n) % padded;
  uint my_hist = wid / splinters;

  // load histogram into shared memory
  if (eid < n_groups) {
    local_hist[lid] = global_hist[eid + my_hist * n_groups];
  } else {
    local_hist[lid] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // upsweep
  uint stride = 1;
  while (stride < n) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int index = (lid + 1) * stride * 2 - 1;
    if (index < n) {
      local_hist[index] += local_hist[index - stride];
    }
    stride *= 2;
  }
  if (lid == 0) {
    counts_b[wid] = local_hist[n - 1];
    local_hist[n - 1] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // downsweep
  stride = n / 2;
  while (stride >= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int index = (lid + 1) * stride * 2 - 1;
    if (index < n) {
      uint left = local_hist[index - stride];
      local_hist[index - stride] = local_hist[index];
      local_hist[index] += left;
    }
    stride /= 2;
  }

  // write back shared memory to global
  barrier(CLK_LOCAL_MEM_FENCE);
  if (eid < n_groups) {
    global_hist[eid + my_hist * n_groups] = local_hist[lid];
  }
}

// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum_b(
    __global uint *global_hist, // must be 256 entries per workgroup
    __global uint *counts_b,    // must be splinters*256 many
    uint n_groups, uint splinters) {
  uint n = get_local_size(0);
  uint gid = get_global_id(0);

  if (gid < 256) {
    uint prefix_sum = 0;
    for (uint i = 0; i < splinters; i++) {
      uint temp = counts_b[i + gid * splinters];
      counts_b[i + gid * splinters] = prefix_sum;
      prefix_sum += temp;
    }
  }
}

// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum_c(
    __global uint *global_hist, // must be 256 entries per workgroup
    __global uint *counts_b,    // must be splinters*256 many
    uint n_groups, uint splinters) {
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  uint gid = get_global_id(0);

  uint padded = splinters * n;
  uint eid = (lid + wid * n) % padded;
  uint my_hist = wid / splinters;

  // load histogram into shared memory
  if (eid < n_groups) {
    atomic_add(&global_hist[eid + my_hist * n_groups], counts_b[wid]);
  }
}

// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_reorder(
    __global uint2 *input,      // must be of size n
    __global uint2 *output,     // must be of size n
    __global uint *global_hist, // must be 256 entries per workgroup
    __global uint *counts,      // must be 256 entries
    __local uint *local_hist,   // must be of size 256 for each workgroup
    uint shift,                 // must be one of [0,8,16,24]
    uint n) {
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n_groups = get_num_groups(0);

  // write global hist back to local hist
  local_hist[lid] = global_hist[lid * n_groups + wid] + counts[lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // sort output based on counts and global hist
  if (gid < n && lid == 0) {
    for (uint i = 0; i < 256; i++) {
      uint j = gid + i;
      if (j < n) {
        uint key = ((input[j].x) >> shift) & 255;
        uint pos = atomic_inc(&local_hist[key]);
        output[pos] = input[j];
      }
    }
  }
}

// ~~~~~~~~~~~~ RESORT POS, VEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__kernel void resort_data_a(__global uint2 *handles, __global const float2 *pos,
                            __global const float2 *vel,
                            __global float2 *pos_buf, __global float2 *vel_buf,
                            // constants
                            uint n) {
  int i = get_global_id(0); // current thread
  if (i >= n) {
    return;
  }
  pos_buf[i] = pos[handles[i].y];
  vel_buf[i] = vel[handles[i].y];
}

__kernel void resort_data_b(__global uint2 *handles, __global float2 *pos,
                            __global float2 *vel,
                            __global const float2 *pos_buf,
                            __global const float2 *vel_buf,
                            // constants
                            uint n) {
  int i = get_global_id(0); // current thread
  if (i >= n) {
    return;
  }
  pos[i] = pos_buf[i];
  vel[i] = vel_buf[i];
  handles[i].y = i;
}

// ~~~~~~~~~~~~ REDUCE  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// as described in:
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__kernel void reduce_min(__global float2 *arr, __global float2 *res,
                         __local float2 *l_arr,
                         // arguments
                         uint n, uint is_first_pass, float2 upper_min_limit) {
  uint gid = get_global_id(0);
  uint wid = get_group_id(0);
  uint lid = get_local_id(0);
  uint local_size = get_local_size(0);

  // load array into local memory
  if (gid < n) {
    l_arr[lid] =
        min(((is_first_pass == 0) ? res[gid] : arr[gid]), upper_min_limit);
  } else {
    l_arr[lid] = upper_min_limit;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // perform local reduction
  for (uint s = local_size / 2; s > 0; s /= 2) {
    if (lid < s) {
      l_arr[lid] = min(l_arr[lid], l_arr[lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write back the result
  if (lid == 0) {
    res[wid] = l_arr[0];
  }
}

__kernel void reduce_max_magnitude(__global float2 *arr, __global float *res,
                                   __local float *l_arr,
                                   // arguments
                                   uint n, uint is_first_pass) {
  uint gid = get_global_id(0);
  uint wid = get_group_id(0);
  uint lid = get_local_id(0);
  uint local_size = get_local_size(0);

  // load array into local memory
  if (gid < n) {
    l_arr[lid] = ((is_first_pass == 0) ? length(res[gid]) : length(arr[gid]));
  } else {
    l_arr[lid] = FLT_MIN;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // perform local reduction
  for (uint s = local_size / 2; s > 0; s /= 2) {
    if (lid < s) {
      l_arr[lid] = max(l_arr[lid], l_arr[lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write back the result
  if (lid == 0) {
    res[wid] = l_arr[0];
  }
}

__kernel void update_dt(__global float *dt, __global float *t_current,
                        __global float *speed_max, float lambda, float max_dt,
                        float init_dt,
                        // constants
                        float v_eps, float h) {
  if (get_global_id(0) == 0) {
    float new_dt = min(lambda * h / speed_max[0], max_dt);
    if (isnan(new_dt)) {
      new_dt = init_dt;
    }
    t_current[0] += new_dt;
    dt[0] = new_dt;
  }
}

// ~~~~~~~~~~~~ RENDER  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const uchar3 BLUE = (uchar3)(0, 0, 255);
const uchar3 RED = (uchar3)(255, 0, 0);
const uchar3 BLACK = (uchar3)(0, 0, 0);
const uchar3 WHITE = (uchar3)(255, 255, 255);

/// see https://github.com/adammaj1/1D-RGB-color-gradient/blob/main/src/p.c#L197
inline uchar3 magma(float pos) {
  float x, x2, x3, x4, x5, x6, x7, x8;
  float R, G, B;
  x = pos;
  x2 = x * x;
  x3 = x * x2;
  x4 = x * x3;
  x5 = x * x4;
  x6 = x * x5;
  x7 = x * x6;
  x8 = x * x7;
  R = -2.1104070317295411e-002 + 1.0825531148278227e+000 * x -
      7.2556742716785472e-002 * x2 + 6.1700693562312701e+000 * x3 -
      1.1408475082678258e+001 * x4 + 5.2341915705822935e+000 * x5;
  if (R < 0.0)
    R = 0.0;
  G = (-9.6293819919380796e-003 + 8.1951407027674095e-001 * x -
       2.9094991522336970e+000 * x2 + 5.4475501043849874e+000 * x3 -
       2.3446957347481536e+000 * x4);
  if (G < 0.0)
    G = 0.0;
  B = 3.4861713828180638e-002 - 5.4531128070732215e-001 * x +
      4.9397985434515761e+001 * x2 - 3.4537272622690250e+002 * x3 +
      1.1644865375431577e+003 * x4 - 2.2241373781645634e+003 * x5 +
      2.4245808412415154e+003 * x6 - 1.3968425226952077e+003 * x7 +
      3.2914755310075969e+002 * x8;
  return (uchar3)((unsigned char)255 * R, (unsigned char)255 * G,
                  (unsigned char)255 * B);
}
const float2 BORDER_OFFSET = (float2)(0.1, 0.1);
inline bool inbounds(float2 p, float2 low, float2 high) {
  return (p.x > low.x) && (p.x < high.x) && (p.y > low.y) && (p.y < high.y);
}

__kernel void render_image(__global uchar3 *image, __global float2 *pos,
                           __global float2 *vel, __global float *den,
                           __global float2 *pos_min, __global uint2 *handles,
                           uint2 resolution, float world_height, uint n,
                           float ks, float2 b_min, float2 b_max, float alpha,
                           float h, float mass, float rho_zero) {
  // return if out of bounds
  uint gid = get_global_id(0);
  if (gid >= resolution[0] * resolution[1]) {
    return;
  }

  // get world coordinates of current pixel
  float2 pixel_coord = (float2)(gid % resolution[0], gid / resolution[0]);
  float2 normal_coord =
      2.0f * (pixel_coord / ((float2)(resolution[0], resolution[1])) -
              (float2)(0.5, 0.5));
  float aspect = (float)resolution[0] / (float)resolution[1];
  float2 p = (normal_coord * world_height * ((float2)(aspect, -1.0)));

  float speed = 0.0;
  float2 rows_nearby[3] = {
      (float2)(-ks, -ks),
      (float2)(-ks, 0),
      (float2)(-ks, ks),
  };
  bool hit = false;
  for (int k = 0; k < 3; k++) {
    // update neighbour set
    uint key = cell_key(p + rows_nearby[k], ks, pos_min[0] - (float2)(ks, ks));
    int j = binary_search(handles, key, n);
    if (j >= 0) {
      int initial_cell = (handles[j][0]);
      while (j < n && (handles[j][0]) < initial_cell + 3) {
        uint j_i = handles[j][1];
        hit |= distance(pos[j_i], p) <= h;
        speed += length(vel[j_i]) * w(p, pos[j_i], alpha, h) / den[j_i];
        j++;
      }
    }
  }
  speed *= mass;
  float t = max(min(0.1f + speed / 20.0f, 1.0f), 0.0f);
  uchar3 colour = magma(t);

  image[gid] = (!inbounds(p, b_min, b_max) &&
                inbounds(p, b_min - BORDER_OFFSET, b_max + BORDER_OFFSET))
                   ? WHITE
                   : (hit ? colour : BLACK);
}
