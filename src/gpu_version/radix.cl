
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

// assumes that both the local and global worksize are 256!
__kernel void radix_sort_b(
  __global uint* global_hist, // must be 256 entries per workgroup
  __global uint* counts, // must be 256 entries
  uint n_256
){
  uint gid = get_global_id(0);
  uint n_groups = n_256/get_local_size(0);

  // compute exclusive prefix sum of counts
  if(gid==0){
    uint prefix_sum = 0;
    for(uint i=0; i<256; i++){
      uint temp = counts[i];
      counts[i] = prefix_sum;
      prefix_sum += temp;
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(gid < 256){
    uint prefix_sum = 0;
    for (uint i=0; i<n_groups; i++){
      uint temp = global_hist[i+gid*n_groups];
      global_hist[i+gid*n_groups] = prefix_sum;
      prefix_sum += temp;
    }
  }
}


// assumes local worksize is n_groups
__kernel void radix_sort_prefixsum(
  __global uint* global_hist, // must be 256 entries per workgroup
  __local uint* local_hist, // must be n_groups large
  __global uint* counts, // must be 256 entries
  uint n_groups
){
  uint gid = get_global_id(0);
  if(gid==0){
    uint prefix_sum = 0;
    for(uint i=0; i<256; i++){
      uint temp = counts[i];
      counts[i] = prefix_sum;
      prefix_sum += temp;
    }
  }

  // compute exclusive prefix sum of all histograms
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n = get_local_size(0);
  
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
