// Copyright 2023 Pierre Talbot

#include <cstdio>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>

#ifndef UTILITY_HPP
#define UTILITY_HPP

#define CUDIE(result) { \
  cudaError_t e = (result); \
  if (e != cudaSuccess) { \
    printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }}

/** Initialize a float vector of size n*/
std::vector<int> init_float_vector(size_t n, std::mt19937& m) {
  std::vector<int> v(n);
  // CUDIE(cudaMallocManaged(&v, sizeof(float) * n));
  // std::mt19937 m{std::random_device{}()};
  // std::mt19937 m{0}; // fixed seed to ease debugging
  std::uniform_int_distribution<int> dist{1, 99};
  for(int i = 0; i < v.size(); ++i) {
    v[i] = dist(m);
  }
  return std::move(v);
}

/** Copy a CPU vector to the managed memory of the GPU. */
template <class T>
T* initialize_gpu_vector(const std::vector<T>& vec) {
  size_t n = vec.size();
  T* gpu_vec;
  // CUDIE(cudaMallocManaged(&gpu_vec, sizeof(T*) * n));
  CUDIE(cudaMalloc((void**)&gpu_vec, sizeof(int) * n));

  // for(int i = 0; i < n; ++i) {
  //   gpu_vec[i] = vec[i];
  // }
  cudaMemcpy(gpu_vec, vec.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
  return gpu_vec;
}


/** Compare two vectors to ensure they are equal. */
template <class T>
void check_equal_vectors(const std::vector<T>& cpu_vec, T* gpu_vec) {
  for(size_t i = 0; i < cpu_vec.size(); ++i) {
    if(cpu_vec[i] != gpu_vec[i]) {
      printf(
        "Found an error: %f != %f\n", cpu_vec[i], gpu_vec[i]
      );
      exit(1);
    }
  }
}


template <class T>
void check_equal_vectorss(const std::vector<T>& cpu_vec, std::vector<T>& gpu_vec) {
  for(size_t i = 0; i < cpu_vec.size(); ++i) {
    if(cpu_vec[i] != gpu_vec[i]) {
      printf(
        "Found an error: %d != %d\n", cpu_vec[i], gpu_vec[i]
      );
      exit(1);
    }
  }
}


/** Benchmarks the time taken by the function `f` by executing it 1 time first for warm-up, then 10 times in a row, then dividing the result by 10.
 * It returns the duration in milliseconds. */
template<class F>
long benchmark_ms(F&& f) {
  f(); // warm-up.
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < 10; ++i) {
    f();
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10;
}

#endif
