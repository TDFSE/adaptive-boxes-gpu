
#include <stdlib.h>
#include <iostream>
// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
//STL
#include <vector>
// cuda call
#include "./include/cuda_call.h"
// kernels
#include "./include/rectangular_explorer_kernel.h"
#include "./include/rectangular_remover_kernel.h"
// rectangle struct
#include "./include/rectangle.h"
// csv
#include "./include/csv_tools.h"
#include "./include/io_tools.h"
// partition system
#include "./include/partition_graph.h"
#include "./include/partition_kernels.h"


int main(int argc, char *argv[]){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	if (argc < 4){
		printf("Error Args: 4 Needed \n[1]input file(binary matrix in .csv)\n[2]output file(list of rectangles in .csv) \n[3]n (# of tests = n*n)\n");
		return 0;
	}

	// Arguments
	std::string input_file_name = argv[1];
	std::string output_file_name = argv[2];
	int n_tests = atoi(argv[3]);


//	 Reading data
    printf("Reading Data...\n");
    std::cout << "reading: " << input_file_name << std::endl;

    csv_data_t csv_data;
    read_numerical_csv(input_file_name, false, csv_data);
//    csv_data.print_data();

	long m = csv_data.m;
	long n = csv_data.n;
    int *data = &csv_data.data_vec[0];
	printf("Data on Memory: Data size: m %ld , n% ld\n",m, n);
	
	// CUDA timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Rectangles vector
	std::vector<rectangle_t> recs;

	// CUDA
	int grid_x = n_tests; // fixed
	int grid_y = n_tests; //
	printf("Number of tests: %d \n",grid_x*grid_y);
	
	// GPU data
	int *data_d;
	int *areas_d;
	int *out_d;

	// Thrust Data
	thrust::device_vector<int> t_data_d(m*n);	
	data_d = thrust::raw_pointer_cast(&t_data_d[0]);

	thrust::device_vector<int> t_areas_d(grid_x*grid_y);
	areas_d = thrust::raw_pointer_cast(&t_areas_d[0]);

	thrust::device_vector<int> t_out_d(grid_x*grid_y*4);
	out_d = thrust::raw_pointer_cast(&t_out_d[0]);	
	
	// Copy data to device memory
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);
	
	// Grid and Block size
	dim3 grid(grid_x, grid_y, 1);
	dim3 block(4, 1, 1); // fixed size
	
	// Init algorithm -----------------------
	cudaEventRecord(start);
	// Setup cuRand
	curandState *devStates;
	CC(cudaMalloc((void **)&devStates, grid_x*grid_y*sizeof(unsigned int)));
	
	setup_kernel<<<grid, block>>>(devStates);
	cudaDeviceSynchronize();

	// Partition system initialization
	const int partition_size = 32;  // Default 32x32 tiles
	const float density_threshold = 0.1f;  // 10% filled pixels threshold
	
	int partition_count = calculate_partition_count(m, n, partition_size);
	printf("Initializing partition system: %d partitions of size %dx%d\n", partition_count, partition_size, partition_size);
	
	// Allocate partition device memory
	partition_t *partitions_d;
	int *adjacency_matrix_d;
	CC(cudaMalloc((void**)&partitions_d, partition_count * sizeof(partition_t)));
	CC(cudaMalloc((void**)&adjacency_matrix_d, partition_count * partition_count * sizeof(int)));
	
	// Initialize partitions with zero values
	CC(cudaMemset(partitions_d, 0, partition_count * sizeof(partition_t)));
	CC(cudaMemset(adjacency_matrix_d, 0, partition_count * partition_count * sizeof(int)));
	
	// Setup partition kernels grid/block dimensions
	dim3 partition_grid(partition_count, 1, 1);
	dim3 partition_block(min(256, partition_size * partition_size), 1, 1);  // Max 256 threads per block
	
	dim3 connectivity_grid((partition_count + 255) / 256, 1, 1);
	dim3 connectivity_block(256, 1, 1);
	
	// Calculate initial density and connectivity
	compute_partition_density<<<partition_grid, partition_block>>>(data_d, m, n, partitions_d, partition_count, partition_size);
	cudaDeviceSynchronize();
	
	build_connectivity_graph<<<connectivity_grid, connectivity_block>>>(partitions_d, partition_count, adjacency_matrix_d, density_threshold, partition_size, n);
	cudaDeviceSynchronize();
	
	update_partition_priorities<<<connectivity_grid, connectivity_block>>>(partitions_d, partition_count, adjacency_matrix_d);
	cudaDeviceSynchronize();

	// Loop
	printf("Working...\n");
	rectangle_t rec;
	int max_step = 999999;
	int sum;
	
	// init last sum
	int last_sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
	
	int last_x1 = -1;
	int last_x2 = -1;
	int last_y1 = -1;
	int last_y2 = -1;
	
	int x1,x2,y1,y2;

	for (int step=0; step<max_step; step++){
		find_largest_rectangle<<<grid,block>>>(devStates,m,n,data_d,out_d, areas_d, partitions_d, partition_count);
		cudaDeviceSynchronize();
		
		thrust::device_vector<int>::iterator iter = thrust::max_element(t_areas_d.begin(), t_areas_d.end());
		unsigned int position = iter - t_areas_d.begin();
		int max_val = *iter; 
			
		if (max_val==0){
			continue;
		}

		x1 = t_out_d[position*4 + 0];  
		x2 = t_out_d[position*4 + 1];  
		y1 = t_out_d[position*4 + 2];  
		y2 = t_out_d[position*4 + 3];  


		if (!((last_x1==x1) & (last_x2==x2) & (last_y1==y1) & (last_y2==y2)) ){
			int dist_y = (y2 - y1) + 1;
			int dist_x = (x2 - x1) + 1;
			int x_blocks = (int)ceil((double)dist_x/2.0);
			int y_blocks = (int)ceil((double)dist_y/2.0);
			
			dim3 tmp_block(2, 2, 1);
			dim3 tmp_grid(x_blocks, y_blocks, 1);

			remove_rectangle_from_matrix<<<tmp_grid, tmp_block>>>(x1,x2,y1,y2, data_d, m, n);
			cudaDeviceSynchronize();
			
			sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
			
			if(sum < last_sum){
				rec.x1 = x1;
				rec.x2 = x2;
				rec.y1 = y1;
				rec.y2 = y2;
				recs.push_back(rec);
			}
			
			last_sum = sum;
			
			// Update partitions every 10th rectangle removal (reduces overhead while maintaining guidance)
			if (step % 10 == 0) {
				// Update affected partitions after rectangle removal
				update_affected_partitions<<<partition_grid, partition_block>>>(x1, x2, y1, y2, partitions_d, partition_count, data_d, m, n, partition_size);
				cudaDeviceSynchronize();
				
				// Recompute priorities for updated partitions
				update_partition_priorities<<<connectivity_grid, connectivity_block>>>(partitions_d, partition_count, adjacency_matrix_d);
				cudaDeviceSynchronize();
			}
			
			/*printf("sum = %d\n", sum);			*/

			if(sum<=0){
				break;
			}
			
			last_x1 = x1;
			last_x2 = x2;
			last_y1 = y1;
			last_y2 = y2;	
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Decomposition ready!!\n");
	printf("-->Elapsed time: %f\n", milliseconds);
	printf("-->Last sum %d\n",sum);
	
	
	/*Saving data in csv format*/
	std::cout << "Saving rectagles -  total amount of rectangles: "<< recs.size() << std::endl;
	save_rectangles_in_csv(output_file_name, &recs);	
	
	// Free memory
	cudaFree(devStates);
	cudaFree(partitions_d);
	cudaFree(adjacency_matrix_d);
	/*delete data;*/

	return 0;
}
