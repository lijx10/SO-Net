#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>


__global__ void index_max_forward_cuda_kernel(const float* __restrict__ data,
		const int* __restrict__ index,
		int* __restrict__ max_idx,
		float* __restrict__ max_val,
		const int B, const int C, const int N, const int K){
	int b = threadIdx.x;
	int c = blockIdx.x;

	for(int n=0;n<N;++n){
		int k = index[b*N+n];
		float data_point = data[b*C*N+c*N+n];
		if (data_point > max_val[b*C*K+c*K+k]){
			max_val[b*C*K+c*K+k] = data_point;
			max_idx[b*C*K+c*K+k] = n;
		}
	}
}



__global__ void index_max_forward_cuda_kernel_shared_mem(const float* __restrict__ data,
		const int* __restrict__ index,
		int* __restrict__ max_idx,
		float* __restrict__ max_val,
		const int B, const int C, const int N, const int K){
	int b = threadIdx.x;
	int c = blockIdx.x;

	extern __shared__ float max_val_shared[];
	for (int i=0;i<K;i+=1){
		max_val_shared[b*K+i] = -1000;
	}

	for(int n=0;n<N;++n){
		int k = index[b*N+n];
		float data_point = data[b*C*N+c*N+n];
		if (data_point > max_val_shared[b*K+k]){
			max_val_shared[b*K+k] = data_point;
			max_idx[b*C*K+c*K+k] = n;
		}
	}
//
//	__syncthreads();

//	for(int n=0;n<N;++n){
//		int k = index[b*N+n];
//		float data_point = data[b*C*N+c*N+n];
//		if (data_point > max_val[b*C*K+c*K+k]){
//			max_val[b*C*K+c*K+k] = data_point;
//			max_idx[b*C*K+c*K+k] = n;
//		}
//	}
}



torch::Tensor index_max_forward_cuda(const torch::Tensor data, const torch::Tensor index, const int K){
	int B = data.size(0);
	int C = data.size(1);
	int N = data.size(2);

	auto device_idx = data.device().index();
        auto max_idx = torch::zeros({B, C, K}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_idx));
        auto max_val = torch::ones({B, C, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_idx)) * -1000.0;

	index_max_forward_cuda_kernel<<<C, B>>>(data.data<float>(),
			index.data<int>(),
			max_idx.data<int>(),
			max_val.data<float>(),
			B, C, N, K);

	return max_idx;
}

torch::Tensor index_max_forward_cuda_shared_mem(const torch::Tensor data, const torch::Tensor index, const int K){
	int B = data.size(0);
	int C = data.size(1);
	int N = data.size(2);

	auto device_idx = data.device().index();
        auto max_idx = torch::zeros({B, C, K}, torch::TensorOptions({torch::kCUDA, device_idx}).dtype(torch::kInt32));
        auto max_val = torch::ones({B, C, K}, torch::TensorOptions({torch::kCUDA, device_idx}).dtype(torch::kFloat32)) * -1000.0;

	index_max_forward_cuda_kernel_shared_mem<<<C, B, B*K*sizeof(float)>>>(data.data<float>(),
			index.data<int>(),
			max_idx.data<int>(),
			max_val.data<float>(),
			B, C, N, K);

	return max_idx;
}
