#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <thread>

// cpu operations -------------------------------
void max_forward_worker(torch::TensorAccessor<float, 3>* p_data_a,
                torch::TensorAccessor<int, 2>* p_index_a,
                torch::TensorAccessor<int, 3>* p_max_idx_a,
                torch::TensorAccessor<float, 3>* p_max_val_a,
		int c_begin, int c_end) {
	int B = p_data_a->size(0);
	// int C = p_data_a->size(1);
	int N = p_data_a->size(2);
	// int K = p_max_idx_a->size(2);

	// thread is on C channel
	for (int b=0; b<B; ++b) {
		for (int c=c_begin; c<c_end; ++c) {
			for (int n=0; n<N; ++n) {
				int k = (*p_index_a)[b][n];
				float data_point = (*p_data_a)[b][c][n];
				if (data_point > (*p_max_val_a)[b][c][k]) {
					(*p_max_val_a)[b][c][k] = data_point;
					(*p_max_idx_a)[b][c][k] = n;
				}
			}
		}
	}
}

torch::Tensor index_max_forward_pthread_cpu(const torch::Tensor data,
                const torch::Tensor index,
		const int K,
		const int thread_num) {
	int B = data.size(0);
	int C = data.size(1);
	// int N = data.size(2);

        auto max_idx = torch::zeros({B, C, K}, torch::TensorOptions().dtype(torch::kInt32));
        auto max_val = torch::ones({B, C, K}, torch::TensorOptions().dtype(torch::kFloat32)) * -1000.0;

	// use accessor
        auto data_a = data.accessor<float, 3>();
        auto index_a = index.accessor<int, 2>();
	auto max_idx_a = max_idx.accessor<int, 3>();
	auto max_val_a = max_val.accessor<float, 3>();

	// multi thread for loop, divide on C channel
	std::thread thread_pool[thread_num];
	int c_interval = int(C) / int(thread_num);
	for (int t=0;t<thread_num;++t) {
		int c_begin = t*c_interval;
		int c_end = 0;
		if (t==thread_num-1){
			c_end = C;
		}
		else{
			c_end = (t+1) * c_interval;
		}

		thread_pool[t] = std::thread(max_forward_worker, &data_a, &index_a, &max_idx_a, &max_val_a, c_begin, c_end);
	}
	for (int t=0;t<thread_num;++t) {
		thread_pool[t].join();
	}

        return max_idx;
}


torch::Tensor index_max_forward_cpu(const torch::Tensor data,
                const torch::Tensor index,
		const int K) {
	int B = data.size(0);
	int C = data.size(1);
	int N = data.size(2);

        torch::Tensor max_idx = torch::zeros({B, C, K}, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
        torch::Tensor max_val = torch::ones({B, C, K}, torch::TensorOptions().dtype(torch::kFloat32)) * -1000.0;

	// conversion between tensor and variable
//	std::cout<<max_idx.type()<<std::endl;
//	std::cout<<torch::autograd::make_variable(max_idx, false).type()<<std::endl;
//	std::cout<<torch::autograd::make_variable(max_idx, false).data().type()<<std::endl;
//	std::cout<<data.type()<<std::endl;
//	std::cout<<torch::autograd::make_variable(data.data(), false).type()<<std::endl;
//	std::cout<<torch::autograd::make_variable(data.data(), false).data().type()<<std::endl;

        // use accessor
        auto data_a = data.accessor<float, 3>();
        auto index_a = index.accessor<int, 2>();
        auto max_idx_a = max_idx.accessor<int, 3>();
        auto max_val_a = max_val.accessor<float, 3>();

	// single thread for loop
	for (int b=0; b<B; ++b) {
		for (int c=0; c<C; ++c) {
			for (int n=0; n<N; ++n) {
				int k = index_a[b][n];
				float data_point = data_a[b][c][n];
				if (data_point > max_val_a[b][c][k]) {
					max_val_a[b][c][k] = data_point;
					max_idx_a[b][c][k] = n;
				}
			}
		}
	}

        return max_idx;
}
// cpu operations -------------------------------




// cuda operations ------------------------------
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor/variable")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// declare the functions in .cu file
torch::Tensor index_max_forward_cuda(const torch::Tensor data,
                const torch::Tensor index,
		const int K);

torch::Tensor index_max_forward_cuda_shared_mem(const torch::Tensor data,
                const torch::Tensor index,
		const int K);

torch::Tensor index_max_forward_cuda_wrapper(const torch::Tensor data,
                const torch::Tensor index,
		const int K){
	CHECK_INPUT(data);
	CHECK_INPUT(index);

        return index_max_forward_cuda(data, index, K);
}

torch::Tensor index_max_forward_cuda_wrapper_shared_mem(const torch::Tensor data,
                const torch::Tensor index,
		const int K){
	CHECK_INPUT(data);
	CHECK_INPUT(index);

        return index_max_forward_cuda_shared_mem(data, index, K);
}
// cuda operations ------------------------------




PYBIND11_MODULE(index_max, m) {
	m.def("forward_cpu", &index_max_forward_cpu, "CPU single thread");
	m.def("forward_multi_thread_cpu", &index_max_forward_pthread_cpu, "CPU multi-thread");
        m.def("forward_cuda", &index_max_forward_cuda_wrapper, "CUDA code without shared memory");
        m.def("forward_cuda_shared_mem", &index_max_forward_cuda_wrapper_shared_mem, "CUDA code with shared memory");
}
