extern "C" __global__ void myop_kernel_impl(const float* x, float* y, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}


