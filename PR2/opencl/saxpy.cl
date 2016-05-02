// SAXPY (Single precision real Alpha X plus Y)
// Original source: Banger, R, Bhattacharyya .K. OpenCL Programming by Example. 2013. Packt publishing
// By: yohanes.gultom@gmail.com

__kernel void saxpy_kernel(float alpha, __global float *A, __global float *B, __global float *C)
{
  //Get the index of the work-item
  int index = get_global_id(0);
  C[index] = alpha* A[index] + B[index];
}
