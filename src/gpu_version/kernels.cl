


// ~~~~~~~~~~~~ KERNEL FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// cubic spline kernel function
float w(float2 x_i, float2 x_j, float alpha, float h){
  float q = distance(x_i,x_j)/h;
  float t1 = max(0.0, 1.0-q);
  float t2 = max(0.0, 2.0-q);
  return alpha * (t2*t2*t2 - 4.0*t1*t1*t1);
}

float2 dw(float2 x_i, float2 x_j, float alpha, float h){
  float q = distance(x_i,x_j)/h;
  float t1 = max(0.0, 1.0-q);
  float t2 = max(0.0, 2.0-q);
  float magnitude = (alpha/h) * (-3.0*t2*t2 + 12.0*t1*t1);
  return (x_i-x_j) * magnitude;
}


// ~~~~~~~~~~~~ SPH APPROXIMATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__kernel void eulercromerstep(
  // arrays
  __global float2* pos, 
  __global float2* vel, 
  __global float2* acc, 
  // atomics/variables
  float dt
){
  int i = get_global_id(0);
  // float2 new_v = vel[i] + acc[i]*dt;
  vel[i] += acc[i]*dt;
  pos[i] += vel[i]*dt;
}

__kernel void enforce_boundary(
  // arrays
  __global float2* pos, 
  __global float2* vel, 
  __global float2* acc, 
  // constants
  float2 min, 
  float2 max
) {
  int i = get_global_id(0);
  if(pos[i].x<=min.x){pos[i].x=min.x; acc[i].x=0.0; vel[i].x=0.0;}
  if(pos[i].y<=min.y){pos[i].y=min.y; acc[i].y=0.0; vel[i].y=0.0;}
  if(pos[i].x>=max.x){pos[i].x=max.x; acc[i].x=0.0; vel[i].x=0.0;}
  if(pos[i].y>=max.y){pos[i].y=max.y; acc[i].y=0.0; vel[i].y=0.0;}
}

/// n is the number of particles
/// cells and indices are the two components of the handles-array
/// neighbours is a flat array of chunks of 3, containing indices
///  into the handles array
__kernel void update_densities_pressures(
  // atomics
  float k,
  float rho_0,
  // arrays
  __global float* den,
  __global float* prs,
  __global float2* pos, 
  __global uint2* handles,
  __global int* neighbours,
  // constants
  float mass,
  float alpha,
  float h,
  uint n
){
  int i = get_global_id(0);
  float new_den = 0.0;
  float2 p = pos[i];

  // sum over neighbours
  for (int k=0; k<3; k++){
    int j=neighbours[3*i+k]; // j is an index into `handles`
    if (j>=0){
      int initial_cell = (handles[j][0]);
      while (j<n && (handles[j][0])<initial_cell+3){
        new_den += w(p, pos[handles[j][1]], alpha, h);
        j++;
      }
    }
  }

  new_den *= mass;
  den[i] = new_den;
  prs[i] = max(0.0, k*(new_den/rho_0-1.));
}

__kernel void apply_gravity_viscosity(
  // atomics
  float nu,
  float g,
  // arrays
  __global float2* acc,
  __global float2* vel,
  __global float2* pos,
  __global float* den,
  __global uint2* handles,
  __global int* neighbours,
  // constants
  float mass,
  float alpha,
  float h,
  uint n
){
  int i = get_global_id(0);
  float2 acc_g = (float2)(0.0, g);
  float2 vis = (float2)(0.0, 0.0);
  float2 x_i = pos[i];
  float2 v_i = vel[i];

  // sum over neighbours
  for (int k=0; k<3; k++){
    int j=neighbours[3*i+k]; // j is an index into `handles`
    if (j>=0){
      int initial_cell = handles[j][0];
      while (j<n && (handles[j][0])<initial_cell+3){
        float2 x_i_j = x_i-pos[handles[j][1]];
        float2 v_i_j = v_i-vel[handles[j][1]];
        float squared_dist = length(x_i_j)*length(x_i_j);
        vis += (mass/den[handles[j][1]]) * 
          (dot(v_i_j, x_i_j)/(squared_dist + 0.01f*h*h)) * 
          dw(x_i, pos[handles[j][1]], alpha, h);
        j++;
      }
    }
  }
  vis *= nu*2.0f;
  acc[i] = acc_g + vis;
}

__kernel void add_pressure_acceleration(
  // atomics
  float rho_0,
  // arrays
  __global float2* pos, 
  __global float2* acc, 
  __global float* den,
  __global float* prs,
  __global uint2* handles,
  __global int* neighbours,
  // constants
  float mass,
  float alpha,
  float h,
  uint n
){
  int i = get_global_id(0);
  float2 x_i = pos[i];
  float2 force = (float2)(0.0f,0.0f);
  float p_i_over_rho_i_squared = prs[i]/(den[i]*den[i]);

  // sum over neighbours
  for (int k=0; k<3; k++){
    int j=neighbours[3*i+k]; // j is an index into `handles`
    if (j>=0){
      int initial_cell = handles[j][0];
      while (j<n && (handles[j][0])<initial_cell+3){
        uint neighbour = handles[j][1];
        // symmetric formula for pressure forces
        force -= dw(x_i, pos[neighbour], alpha, h) * 
          (p_i_over_rho_i_squared + prs[neighbour]/(den[neighbour]*den[neighbour]));
        j++;
      }
    }
  }

  acc[i] += force*mass;
}

// ~~~~~~~~~~~~ NEIGHBOURHOOD SEARCH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

uint cell_key(float2 p, float ks, float2 min){
  uint2 key = convert_uint2_rtn(convert_ushort2_rtn((p-min)/ks));
  return (key.y <<16) + key.x;
}

__kernel void compute_cell_keys(
  // atomics
  float2 min_extent,
  float ks,
  // arrays
  __global float2* pos, 
  __global uint2* handles
){
  int i = get_global_id(0);
  handles[i].x = cell_key(pos[handles[i].y], ks, min_extent);
}


__kernel void compute_neighbours(
  // atomics
  float2 min_extent,
  // arrays
  __global float2* pos, 
  __global uint2* handles,
  __global int* neighbours,
  // constants 
  float ks,
  uint n
){
  int i = get_global_id(0);
  if(i>= n){return;}
  float2 rows_nearby[3] = {
    (float2)(-ks, -ks),
    (float2)(-ks, 0),
    (float2)(-ks, ks),
  };
  for (int k=0; k<3; k++){
    uint key = cell_key(pos[i]+rows_nearby[k],ks,min_extent);
    // binary search for key
    int result = -1;
    int low = 0;
    int high = n-1;
    while(low<=high){
      int mid = (high-low)/2+low;
      uint hit = handles[mid].x;
      if(
        (key<=hit && hit<key+3) && 
        (
          mid == 0 || 
          !(key<=handles[mid-1].x && handles[mid-1].x<key+3)
        )
      ){
        result = mid;
        break;
      } else {
        if(hit < key){
          low=mid+1;
        } else {
          high = mid-1;
        }
      }
    }
    // update neighbour set 
    neighbours[i*3+k] = result;
  }
}

// ~~~~~~~~~~~~ RADIX SORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_histograms(
  __global uint2* input, // must be of size n
  __global uint2* output, // must be of size n
  __global uint* global_hist, // must be 256 entries per workgroup
  __global uint* counts, // must be 256 entries
  __local uint* local_hist, // must be of size 256 for each workgroup
  uint shift, // must be one of [0,8,16,24]
  uint n
){
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n_groups = get_num_groups(0);

  // initialize local histogram to zeros
  local_hist[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  // calculate key using selected byte and enter it 
  // in the local histogram
  if(gid < n){
    uint key = ((input[gid].x) >> shift) & 255;
    atomic_inc(&local_hist[key]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // write back histograms to global memory to compute the global prefix sum
  global_hist[lid*n_groups + wid] = local_hist[lid];
  atomic_add(&counts[lid], local_hist[lid]);
}


// USE FOR ARRAY SIZES LESS THAN 2^18 = 262_144
__kernel void radix_sort_prefixsum_small(
  __global uint* global_hist, // must be 256 entries per workgroup
  __local uint* local_hist, // must be n_groups large
  __global uint* counts, // must be 256 entries
  uint n_groups
){
  // compute exclusive prefix sum of all histograms
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  uint gid = get_global_id(0);
  if(gid==0){
    uint prefix_sum = 0;
    for(uint i=0; i<256; i++){
      uint temp = counts[i];
      counts[i] = prefix_sum;
      prefix_sum += temp;
    }
  }

  // load histogram into shared memory
  if (lid < n_groups){
    local_hist[lid] = global_hist[lid + wid*n_groups];
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
  if (lid < n_groups){
    global_hist[lid + wid*n_groups] = local_hist[lid];
  }
}


// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum_a(
  __global uint* global_hist, // must be 256 entries per workgroup
  __local uint* local_hist, // must be n_groups large
  __global uint* counts, // must be 256 entries
  __global uint* counts_b,  // must be splinters*256 many
  uint n_groups,
  uint splinters
){
  // compute exclusive prefix sum of all histograms
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  uint gid = get_global_id(0);
  if(gid==0){
    uint prefix_sum = 0;
    for(uint i=0; i<256; i++){
      uint temp = counts[i];
      counts[i] = prefix_sum;
      prefix_sum += temp;
    }
  }

  // compute effective id
  uint padded = splinters * n;
  uint eid = (lid+wid*n) % padded;
  uint my_hist = wid/splinters;

  // load histogram into shared memory
  if (eid < n_groups){
    local_hist[lid] = global_hist[eid + my_hist*n_groups];
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
    counts_b[wid] = local_hist[n-1];
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
  if (eid < n_groups){
    global_hist[eid + my_hist*n_groups] = local_hist[lid];
  }
}

// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum_b(
  __global uint* global_hist, // must be 256 entries per workgroup
  __global uint* counts_b, // must be splinters*256 many
  uint n_groups,
  uint splinters
){
  uint n = get_local_size(0);
  uint gid = get_global_id(0);

  if (gid < 256){
    uint prefix_sum = 0;
    for(uint i=0; i<splinters; i++){
      uint temp = counts_b[i+gid*splinters];
      counts_b[i+gid*splinters] = prefix_sum;
      prefix_sum += temp;
    }
  }
}

// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum_c(
  __global uint* global_hist, // must be 256 entries per workgroup
  __global uint* counts_b,  // must be splinters*256 many
  uint n_groups,
  uint splinters
){
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  uint gid = get_global_id(0);

  uint padded = splinters * n;
  uint eid = (lid+wid*n) % padded;
  uint my_hist = wid/splinters;

  // load histogram into shared memory
  if (eid < n_groups){
    atomic_add(&global_hist[eid + my_hist*n_groups], counts_b[wid]);
  } 
}

// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_reorder(
  __global uint2* input, // must be of size n
  __global uint2* output, // must be of size n
  __global uint* global_hist, // must be 256 entries per workgroup
  __global uint* counts, // must be 256 entries
  __local uint* local_hist, // must be of size 256 for each workgroup
  uint shift, // must be one of [0,8,16,24]
  uint n
){
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n_groups = get_num_groups(0);

  // write global hist back to local hist
  local_hist[lid] = global_hist[lid*n_groups + wid] + counts[lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // sort output based on counts and global hist
  if (gid<n && lid == 0){
    for(uint i=0; i<256; i++){
      uint j = gid+i;
      if (j<n){
        uint key = ((input[j].x) >> shift) & 255;
        uint pos = atomic_inc(&local_hist[key]);
        output[pos] = input[j];
      }
    }
  }
}



// ~~~~~~~~~~~~ RESORT POS, VEL ~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
__kernel void resort_data_a(
  __global uint2* handles,
  __global const float2* pos,
  __global const float2* vel,
  __global float2* pos_buf,
  __global float2* vel_buf,
  // constants 
  uint n
){
  int i = get_global_id(0); // current thread
  if(i>= n){return;}
  pos_buf[i] = pos[handles[i].y];
  vel_buf[i] = vel[handles[i].y];
}

__kernel void resort_data_b(
  __global uint2* handles,
  __global float2* pos,
  __global float2* vel,
  __global const float2* pos_buf,
  __global const float2* vel_buf,
  // constants 
  uint n
){
  int i = get_global_id(0); // current thread
  if(i>= n){return;}
  pos[i] = pos_buf[i];
  vel[i] = vel_buf[i];
  handles[i].y = i;
}

