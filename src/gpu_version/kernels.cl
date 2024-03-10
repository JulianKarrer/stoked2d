


// ~~~~~~~~~~~~ KERNEL FUNCTIONS ~~~~~~~~~~~~

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


// ~~~~~~~~~~~~ SPH APPROXIMATIONS ~~~~~~~~~~~~

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

// ~~~~~~~~~~~~ NEIGHBOURHOOD SEARCH ~~~~~~~~~~~~

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

// ~~~~~~~~~~~~ SORTING ALGORITHM ~~~~~~~~~~~~

// http://www.bealto.com/gpu-sorting_parallel-selection-local.html
__kernel void sort_handles_simple(
  __global uint2* in,
  __global uint2* out,
  // constants 
  uint n,
  __local uint* aux
)
{
  int i = get_global_id(0); // current thread
  if(i>= n){return;}
  int wg = get_local_size(0); // workgroup size
  uint2 iData = in[i]; // input record for current thread
  uint iKey = (iData).x; // input key for current thread
  int blockSize = 4 * wg; // block size

  // Compute position of iKey in output
  int pos = 0;
  // Loop on blocks of size BLOCKSIZE keys (BLOCKSIZE must divide N)
  for (int j=0;j<n;j+=blockSize)
  {
    // Load BLOCKSIZE keys using all threads (BLOCK_FACTOR values per thread)
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int index=get_local_id(0);index<blockSize;index+=wg)
      aux[index] = (in[j+index]).x;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop on all values in AUX
    for (int index=0;index<blockSize && j+index<n;index++)
    {
      uint jKey = aux[index]; // broadcasted, local memory
      bool smaller = (jKey < iKey) || ( jKey == iKey && (j+index) < i ); // in[j] < in[i] ?
      pos += (smaller)?1:0;
    }
  }
  out[pos] = iData;
}
// {
//   int i = get_local_id(0); // index in workgroup
//   int wg = get_local_size(0); // workgroup size = block size, power of 2

//   // Move IN, OUT to block start
//   int offset = get_group_id(0) * wg;
//   in += offset; out += offset;

//   // Load block in AUX[WG]
//   aux[i] = in[i];
//   barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

//   // Now we will merge sub-sequences of length 1,2,...,WG/2
//   for (int length=1;length<wg;length<<=1)
//   {
//     uint2 iData = aux[i];
//     uint iKey = iData.x;
//     int ii = i & (length-1);  // index in our sequence in 0..length-1
//     int sibling = (i - ii) ^ length; // beginning of the sibling sequence
//     int pos = 0;
//     for (int inc=length;inc>0;inc>>=1) // increment for dichotomic search
//     {
//       int j = sibling+pos+inc-1;
//       uint jKey = aux[j].x;
//       bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );
//       pos += (smaller)?inc:0;
//       pos = min(pos,length);
//     }
//     int bits = 2*length-1; // mask for destination
//     int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
//     barrier(CLK_LOCAL_MEM_FENCE);
//     aux[dest] = iData;
//     barrier(CLK_LOCAL_MEM_FENCE);
//   }

//   // Write output
//   out[i] = aux[i];
// }



__kernel void copy_handles(
  __global uint2* in,
  __global uint2* out,
  // constants 
  uint n
){
  int i = get_global_id(0); // current thread
  if(i>= n){return;}
  out[i] = in[i];
}

// ~~~~~~~~~~~~ RESORT POS, VEL ~~~~~~~~~~~~
__kernel void resort_data_a(
  __global uint2* handles,
  __global float2* pos,
  __global float2* vel,
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
  __global float2* pos_buf,
  __global float2* vel_buf,
  // constants 
  uint n
){
  int i = get_global_id(0); // current thread
  if(i>= n){return;}
  pos[i] = pos_buf[i];
  vel[i] = vel_buf[i];
  handles[i].y = i;
}

