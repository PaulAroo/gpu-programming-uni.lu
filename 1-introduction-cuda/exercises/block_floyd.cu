// Copyright 2023 Pierre Talbot

#include "../utility.hpp"
#include <string>

void floyd_warshall_cpu(std::vector<std::vector<int>>& d) {
  size_t n = d.size();
  for(int k = 0; k < n; ++k) {
    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < n; ++j) {
        if(d[i][j] > d[i][k] + d[k][j]) {
          d[i][j] = d[i][k] + d[k][j];
        }
      }
    }
  }
}

// Correctness of the algorithm
// For a fixed k:
// • No write conflict: each iteration modifies a different memory cell, namely d[i][j].
// • No read conflict: each iteration reads cell that are not written in, namely d[i][k] and d[k][j].
// • No enforced order: within the two nested loops, the order of the operations does not
// matter.

__global__ void floyd_warshall_gpu(int** d, size_t arr_size) {
  for(int k = 0; k < arr_size; ++k) {
    for(int i = 0; i < arr_size; ++i) {
      for(int j = threadIdx.x; j < arr_size; j += blockDim.x) { 
        //memory access is contiguous (i.e each execution starts out accessing contigous part of the memory)
        if(d[i][j] > d[i][k] + d[k][j]) {
          d[i][j] = d[i][k] + d[k][j];
        }
      }
    }
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: " << argv[0] << " <matrix size> <block size>" << std::endl;
    exit(1);
  }
  size_t n = std::stoi(argv[1]);
  size_t block_size = std::stoi(argv[2]);

  std::cout << "floyd algorithm execution on a single block: Matrix size - " << n << " No of threads - " << block_size << std::endl;

  // I. Generate a random distance matrix of size N x N.
  std::vector<std::vector<int>> cpu_distances = initialize_distances(n);
  // Note that `std::vector` cannot be used on GPU, hence we transfer it into a simple `int**` array in managed memory.
  int** gpu_distances = initialize_gpu_distances(cpu_distances);

  // II. Running Floyd Warshall on CPU.
  long cpu_ms = benchmark_one_ms([&]{
    floyd_warshall_cpu(cpu_distances);
  });
  std::cout << "CPU: " << cpu_ms << " ms" << std::endl;

  // III. Running Floyd Warshall on GPU (single block of size `block_size`).
  long gpu_ms = benchmark_one_ms([&]{
    floyd_warshall_gpu<<<1, block_size>>>(gpu_distances, n);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU: " << gpu_ms << " ms" << std::endl;

  // IV. Verifying both give the same result and deallocating.
  check_equal_matrix(cpu_distances, gpu_distances);
  deallocate_gpu_distances(gpu_distances, n);
  return 0;
}
