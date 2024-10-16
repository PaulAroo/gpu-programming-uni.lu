// Copyright 2023 Pierre Talbot

#include "./utility.hpp"
#include <stdlib.h>
#include <cstdio>

void scan_sequential(int* arr, int* new_arr, int identity_element, size_t arr_size) {
  new_arr[0] = identity_element;

  for(size_t i = 1; i < arr_size; ++i) {
    new_arr[i] = arr[i - 1] + new_arr[i];
  }

}


__global__ void accu_sum(int* arr, int* arr_acc, size_t size) {
  for(size_t i = 0; i < size; ++i) {
    arr_acc[i] = arr_acc[i] + arr[i];
  }
}

int main(int argc, char** argv) {
  // std::vector<int> vec = {1, 2, 3, 4};

  int* arr = (int *)malloc(64 * sizeof(int));
  int* local_scan;

  CUDIE(cudaMallocManaged(&local_scan, sizeof(int) * 64));

  // for(int )

  // accu_sum<<<1, 64>>>();
  cudaDeviceSynchronize();
  return 0;
}
