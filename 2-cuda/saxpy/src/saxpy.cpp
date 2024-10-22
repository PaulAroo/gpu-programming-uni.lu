
#include "utility.hpp"

void saxpy(int scale, const std::vector<int>& X, const std::vector<int>& Y, std::vector<int>& result) {
  const size_t N = result.size();

  for(size_t i = 0; i < N; ++i) {
    result[i] = scale * X[i] + Y[i];
  }
}

__global__ void saxpy_gpu(int scale, const int* X, const int* Y, int* result, size_t arr_size) {
  for(size_t i = threadIdx.x; i < arr_size; i+=blockDim.x) {
    result[i] = scale * X[i] + Y[i];
  }
}

int main(int argc, char** agrv) {
  if(argc != 4) {
    std::cout << "usage: " << agrv[0] << " <array_size>" << " <num_blocks>" << " <num_threads>" << std::endl;
    exit(1);
  }

  std::mt19937 m1{0};
  std::mt19937 m2{1};

  const size_t array_size = std::stoll(agrv[1]);
  const size_t num_blocks = std::stoi(agrv[2]);
  const size_t num_threads = std::stoi(agrv[3]);

  const int scale = 3;
  const std::vector<int> X = init_float_vector(array_size, m1);
  const std::vector<int> Y = init_float_vector(array_size, m2);
  std::vector<int> result(array_size);

  int* X_gpu = initialize_gpu_vector(X);
  int* Y_gpu = initialize_gpu_vector(Y);
  int* result_gpu;
  // CUDIE(cudaMallocManaged(&result_gpu, sizeof(float) * array_size));

  CUDIE(cudaMalloc((void**)&result_gpu, sizeof(int) * array_size));

  std::vector<int> box(array_size);


  long cpu_ms = benchmark_ms([&]() {
    saxpy(scale, X, Y, result);
  });
  std::cout << "CPU: " << cpu_ms << "ms" << std::endl;

  long gpu_ms = benchmark_ms([&]() {
    saxpy_gpu<<<num_blocks, num_threads>>>(
      scale, X_gpu, Y_gpu, result_gpu, array_size
    );
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU: " << gpu_ms << "ms" << std::endl;

  cudaMemcpy(box.data(), result_gpu, sizeof(int) * array_size, cudaMemcpyDeviceToHost);

  check_equal_vectorss(result, box);

  cudaFree(X_gpu);
  cudaFree(Y_gpu);
  cudaFree(result_gpu);

}