
#include <math.h>
// getters
#include "./getters.h"
// random
#include "./random_generator.h"
// partition graph
#include "./partition_graph.h"

// getters
using namespace ng;

 /* 
    FIND THE LARGEST RECTANGLE:
    Kernel used for expand a rectangle as large as possible in the data matrix
    The kernel runs in different blocks, each block with 4 threads.

    Each block needs a random point in the matrix(it uses cuRand). 
    If it doesnt find a valid point, the block sleeps(note: other blocks can work if they find a valid point).
    
    Then, inside of each block the four threads expand the rectangle from the initial random point to all sides:
    [thread 0] right-bottom
    [thread 1] right-top
    [thread 2] left-bottom
    [thread 3] left-top

    Each block has a largest rectangle that could find, in order to get the largest Rectangle, then compute the area and
    stores in areas variable.
 */
__global__ void find_largest_rectangle(curandState *state, long m, long n, int *data_matrix, int *out, int *areas, 
                                       partition_t* partitions, int partition_count){


	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];
	//__shared__ int total_max;

	__shared__ int idx_i;
	__shared__ int idx_j;
	
	__shared__ bool is_sleeping;

	//int i = threadIdx.y;
	int j = threadIdx.x;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;
	
	
	/* GET PRIORITY-GUIDED POINT: Select point from high-priority partition
	 * Each thread gets a different high-priority partition to avoid clustering
	 */
	if(j==0){
	        areas[b_i*b_n + b_j] = 0;

		int id = b_i*b_n + b_j;
		curandState localState = state[id];
		
		unsigned int xx;
		unsigned int yy;
		
		// Priority-guided selection: each thread gets different high-priority partition
		bool found_in_partition = false;
		if (partitions != NULL && partition_count > 0) {
			int thread_id = b_i * gridDim.x + b_j; // Unique thread identifier
			int partition_id = get_priority_guided_partition(partitions, partition_count, thread_id);
			
			// Bounds check partition_id
			if (partition_id >= 0 && partition_id < partition_count) {
				partition_t current_partition = partitions[partition_id];
				
				// Validate partition boundaries
				if (current_partition.x_start >= 0 && current_partition.x_end < n && 
				    current_partition.y_start >= 0 && current_partition.y_end < m &&
				    current_partition.x_end >= current_partition.x_start &&
				    current_partition.y_end >= current_partition.y_start) {
					
					// Sample within high-priority partition
					for(int g=0; g<50; g++){
						xx = curand(&localState);
						yy = curand(&localState);
						
						int partition_width = current_partition.x_end - current_partition.x_start + 1;
						int partition_height = current_partition.y_end - current_partition.y_start + 1;
						
						idx_j = current_partition.x_start + (abs((int)xx) % partition_width);
						idx_i = current_partition.y_start + (abs((int)yy) % partition_height);
						
						if (data_matrix[idx_i*n + idx_j]==1){
							is_sleeping = false;
							found_in_partition = true;
							break;
						}
					}
				}
			}
		}
		
		// Fallback to full matrix search if partition guidance failed
		if (!found_in_partition) {
			for(int g=0; g<100; g++){
				xx = curand(&localState);
				yy = curand(&localState);
				idx_i = abs((int)xx)%(m);	
				idx_j = abs((int)yy)%(n);
				if (data_matrix[idx_i*n + idx_j]==1){
					is_sleeping = false;
					break;
				}else{
					is_sleeping = true;
				}
			}
		}
		
		state[id] = localState;

		//printf("priority_guided: thread_id=%d, found_in_partition=%d, idx_i=%d, idx_j=%d\n", b_i*gridDim.x+b_j, found_in_partition, idx_i, idx_j);
	}
	__syncthreads();
	
	// if sleeping true ,disable block-thread work
	if (!is_sleeping){
		// expand the rectangle
		int results[4] = {0,0,0,0};
		if (j==0){
			get_right_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results);
		}

		if (j==1){
			get_right_top_rectangle(idx_i, idx_j, n, data_matrix, results);
		}

		if (j==2){
			get_left_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results);
		}

		if (j==3){
			get_left_top_rectangle(idx_i, idx_j, n, data_matrix, results);
		}

		coords[j*coords_n + 0] = results[0];
		coords[j*coords_n + 1] = results[1];
		coords[j*coords_n + 2] = results[2];
		coords[j*coords_n + 3] = results[3];

		__syncthreads();

		// merge last rectangles
		if (j==0){
			int a = coords[2*coords_n + 1];
			int b = coords[3*coords_n + 1];
			int pl = a;
			if (b > a){
				pl = b;
			}
			coords[4*coords_n + j] = pl;
			out[b_i*b_n*4 + (4*b_j + j) ] = pl;
		}

		if (j==1){
			int a = coords[0*coords_n + 1];
			int b = coords[1*coords_n + 1];
			int pr = a;
			if (b < a){
				pr = b;
			}
			coords[4*coords_n + j] = pr;
			out[b_i*b_n*4 + (4*b_j + j) ] = pr;
		}

		if (j==2){
			int a = coords[1*coords_n + 3];
			int b = coords[3*coords_n + 3];
			int pt = a;
			if (b > a){
				pt = b;
			}
			coords[4*coords_n + j] = pt;
			out[b_i*b_n*4 + (4*b_j + j) ] = pt;
		}

		if (j==3){
			int a = coords[0*coords_n + 3];
			int b = coords[2*coords_n + 3];
			int pb = a;
			if (b < a){
				pb = b;
			}
			coords[4*coords_n + j] = pb;
			out[b_i*b_n*4 + (4*b_j + j) ] = pb;
		}

		__syncthreads();

		if (j==0){
			int a = abs(coords[coords_n*4 + 0] -  coords[coords_n*4 + 1]) + 1;
			int b = abs(coords[coords_n*4 + 2] -  coords[coords_n*4 + 3]) + 1;
			int area = a*b;
			areas[b_i*b_n + b_j] = area;
		}
	}

}

