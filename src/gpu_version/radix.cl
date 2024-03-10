
// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_a(
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
  global_hist[gid] = local_hist[lid];
  atomic_add(&counts[lid], local_hist[lid]);
}

// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_b(
  __global uint* global_hist, // must be 256 entries per workgroup
  __global uint* counts // must be 256 entries
){
  uint gid = get_global_id(0);
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
    for (uint i=gid; i<get_global_size(0); i+=256){
      uint temp = global_hist[i];
      global_hist[i] = prefix_sum;
      prefix_sum += temp;
    }
  }
}

// assumes that local workgroup size is 256 and global size is a multiple of 256
__kernel void radix_sort_c(
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

  // write global hist back to local hist
  local_hist[lid] = global_hist[gid] + counts[lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // sort output based on counts and global hist
  if (gid<n && lid == 0){
    for(uint i=0; i<256; i++){
      uint j = gid+i;
      uint key = ((input[j].x) >> shift) & 255;
      uint pos = atomic_inc(&local_hist[key]);
      output[pos] = input[j];
    }
  }
}
