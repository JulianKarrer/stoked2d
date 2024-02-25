


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
  __global uint* cells,
  __global uint* indices,
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
      int initial_cell = cells[j];
      while (j<n && cells[j]<initial_cell+3){
        new_den += w(p, pos[indices[j]], alpha, h);
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
  __global uint* cells,
  __global uint* indices,
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
      int initial_cell = cells[j];
      while (j<n && cells[j]<initial_cell+3){
        float2 x_i_j = x_i-pos[indices[j]];
        float2 v_i_j = v_i-vel[indices[j]];
        float squared_dist = length(x_i_j)*length(x_i_j);
        vis += (mass/den[indices[j]]) * 
          (dot(v_i_j, x_i_j)/(squared_dist + 0.01f*h*h)) * 
          dw(x_i, pos[indices[j]], alpha, h);
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
  __global uint* cells,
  __global uint* indices,
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
      int initial_cell = cells[j];
      while (j<n && cells[j]<initial_cell+3){
        // symmetric formula for pressure forces
        force -= dw(x_i, pos[indices[j]], alpha, h) * 
          (p_i_over_rho_i_squared + prs[indices[j]]/(den[indices[j]]*den[indices[j]]));
        j++;
      }
    }
  }

  acc[i] += force*mass;
}

// ~~~~~~~~~~~~ NEIGHBOURHOOD SEARCH ~~~~~~~~~~~~

ulong cell_key(float2 p, float ks, float2 min){
  uint2 key = convert_uint2_rtn((p-min)/ks);
  return ((ulong)key.y)<<16 | key.x;
}

__kernel void compute_cell_keys(
  // atomics
  float kernel_support,
  float2 min_extent,
  // arrays
  __global float2* pos, 
  __global uint* cells
){
  int i = get_global_id(0);
  cells[i] = cell_key(pos[i], kernel_support, min_extent);
}


__kernel void compute_neighbours(
  // atomics
  float2 min_extent,
  // arrays
  __global float2* pos, 
  __global uint* cells,
  __global uint* indices,
  __global int* neighbours,
  // constants 
  float ks,
  uint n
){
  int i = get_global_id(0);
  float2 rows_nearby[3] = {
    (float2)(-ks, -ks),
    (float2)(-ks, 0),
    (float2)(-ks, ks),
  };
  for (int k=0; k<3; k++){
    ulong key = cell_key(pos[i]+rows_nearby[k],ks,min_extent);
    // binary search for key
    int result = -1;
    int low = 0;
    int high = n-1;
    while(low<=high){
      int mid = (high-low)/2+low;
      ulong hit = cells[mid];
      if(
        (key<=hit && hit<key+3) && 
        (
          mid == 0 || 
          !(key<=cells[mid-1] && cells[mid-1]<key+3)
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

__kernel void sort_handles(
  // arrays
  __global uint* cells,
  __global uint* indices,
  // constants 
  uint n
){

}
