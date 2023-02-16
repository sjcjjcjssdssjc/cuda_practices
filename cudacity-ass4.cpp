%%cuda --name student_func.cu

//Udacity HW 4
//Radix Sorting
#define THREADS 1024
#include "utils.h"
#include <thrust/host_vector.h>

__global__ void radix_scan_phase1(unsigned long long* d_in,
                         unsigned int* d_interm,
                         int index,
                         size_t numElems) {
    int tid = threadIdx.x;
    int myid = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared unsigned int sdata[];

    if(myid + tid < numElems)sdata[2 * tid] = (d_in[myid + tid] >> index) & 1;
    else sdata[2 * tid] = 0;
    if(myid + tid + 1 < numElems)sdata[2 * tid + 1] = (d_in[myid + tid + 1] >> index) & 1;
    else sdata[2 * tid + 1] = 0;
    int offset = 1;
    for(int d = THREADS >> 1; d > 0; d >>= 1) {
        if(tid < d) {
            int ai = offset * (2 * tid + 1) - 1; 
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
            // no bound checks
        }
        offset <<= 1;
        __syncthreads();
    }
    if(tid == 0)sdata[THREADS - 1] = 0;
    __syncthreads();
    for(int d = 1; d < THREADS; d <<= 1) {
         __syncthreads();
        offset >>= 1;
        if(tid < d) {
            int ai = offset * (2 * tid + 1) - 1; 
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int tmp = sdata[ai];
            sdata[bi] += tmp;
            sdata[ai] = tmp;
        }
        __syncthreads();
    }
    if(myid + tid < numElems)d_in[myid + tid] = sdata[2 * tid];
    if(myid + tid + 1 < numElems)d_in[myid + tid + 1] = sdata[2 * tid + 1];
    if(tid == (THREADS >>1))d_interm[blockIdx.x] = sdata[2 * tid + 1];

}

__global__ void radix_scan_phase2(unsigned int* d_out,
                         unsigned int* d_interm,
                         size_t numblocks,
                         size_t numElems) {
    int tid = threadIdx.x;
    int myid = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared unsigned int sdata[];

    if(2 * tid < numblocks)sdata[2 * tid] = d_interm[2 * tid];
    else sdata[2 * tid] = 0;
    if(2 * tid + 1 < numblocks)sdata[2 * tid + 1] = d_interm[2 * tid + 1];
    else sdata[2 * tid + 1] = 0;
    int offset = 1;
    for(int d = THREADS >> 1; d > 0; d >>= 1) {
        if(tid < d) {
            int ai = offset * (2 * tid + 1) - 1; 
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
            // no bound checks
        }
        offset <<= 1;
        __syncthreads();
    }
    if(tid == THREADS >> 1)sdata[tid] = 0;
    __syncthreads();
    for(int d = 1; d < THREADS; d <<= 1) {
        offset >>= 1;
        if(tid < d) {
            int ai = offset * (2 * tid + 1) - 1; 
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int tmp = sdata[ai];
            sdata[bi] += tmp;
            sdata[ai] = tmp;
        }
        __syncthreads();
    }
    for(int i = 0; i < numblocks; i++){
        if(2 * tid + i * THREADS < numElems) {
            d_out[2 * tid + i * THREADS] += sdata[i];
        }
        if(2 * tid + 1 + i * THREADS < numElems) {
            d_out[2 * tid + 1 + i * THREADS] += sdata[i];
        }
    }
}

__global__ void move_kernel(unsigned int* d_prefsum, unsigned long long *d_output, 
                                              ,unsigned long long* d_input, unsigned int *dist
                                              ,size_t numblocks, size_t numElems, int i) {
    int tid = threadIdx.x;
    int myid = threadIdx.x + blockDim * blockIdx.x;
    int dest_index = 0;
    if(myid < numElems) {
        if((d_input[myid] >> index) & 1) {
            dest_index = myid + dist[1] - d_prefsum[myid];
        } else dest_index = d_prefsum[i];
    }
    d_output[dest_index] = d_input[myid];
 }

__global__ void copyin_kernel(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned long long* d_inputs,
               const size_t numElems) {
    int myid = threadIdx.x + blockDim.x * blockIdx.x;
    if(myid < numElems) {
        d_inputs[myid] = d_inputPos[myid] + ((unsigned long long)d_inputVals[myid] << 32);
    }
}

__global__ void copyout_kernel(unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               unsigned long long* d_outputs,
               const size_t numElems) {
    int myid = threadIdx.x + blockDim.x * blockIdx.x;
    if(myid < numElems) {
        d_outputPos[myid] = d_outputs[myid] >> 32;
        d_outputVals[myid] = d_outputs[myid];
    }
}
__global__ void hist_kernel(unsigned long long* d_inputs,
                                unsigned int *dist, int index,
                                const size_t numElems) {
    int myid = threadIdx.x + blockDim.x * blockIdx.x;
    if(myid < numElems) {
        int index = (d_inputs[myid] >> index) & 1;
        atomicAdd(&dist[index], 1);
    }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {
  size_t blockdim = 1024;
  size_t blocknum = (numElems + blockdim - 1) / blockdim;

  unsigned long long* d_inputs, *d_outputs;
  cudaMalloc(&d_inputs, numElems * sizeof(unsigned long long));
  cudaMalloc(&d_outputs, numElems * sizeof(unsigned long long));
  copyin_kernel<<<blocknum,blockdim>>>(d_inputVals, d_inputPos, d_inputs, numElems);

  const int batch_size = 2;
  unsigned int *dist;
  cudaMalloc(&dist, batch_size * sizeof(unsigned int));

  unsigned int* d_prefsum, *d_interm;
  cudaMalloc(&d_prefsum, numElems * sizeof(unsigned int));
  cudaMalloc(&d_interm, numblocks * sizeof(unsigned int));

  for(int i = 0; i < 64; i++) {
      cudaMemset(dist, 0, batch_size * sizeof(unsigned int));
      if((i & 1) == 0)hist_kernel(d_inputs, dist, i, numElems);
      else hist_kernel(d_outputs, dist, i, numElems);

      radix_scan_phase1<<<blocknum, blockdim / 2, blockdim>>>(d_prefsum, d_interm, i, numElems);
      radix_scan_phase2<<<1, blockdim / 2, blockdim>>>(d_prefsum, d_interm, blocknum, numElems);
      //do not forget to divide 2
      if((i & 1) == 0)move_kernel<<<blocknum, blockdim>>>(d_prefsum, d_outputs, d_inputs, dist, numblocks, numElems, i);
      else move_kernel<<<blocknum, blockdim>>>(d_prefsum, d_inputs, d_outputs, dist, numblocks, numElems, i);

  }
  copyout_kernel<<<blocknum, blockdim>>>(d_outputVals, d_outputPos, d_inputs, numElems);
  printf("%d\n",numElems);
}