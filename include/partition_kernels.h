#ifndef PARTITION_KERNELS_H
#define PARTITION_KERNELS_H

#include <cuda_runtime.h>
#include "partition_graph.h"

/**
 * Compute density for each partition
 * Each block processes one partition, threads process pixels within partition
 * 
 * Args:
 *     data_matrix: Input binary matrix (m x n)
 *     m: Matrix height
 *     n: Matrix width
 *     partitions: Array of partitions to update
 *     partition_count: Number of partitions
 *     partition_size: Size of each partition tile
 */
__global__ void compute_partition_density(int* data_matrix, long m, long n, 
                                         partition_t* partitions, int partition_count, int partition_size) {
    int partition_id = blockIdx.x;
    if (partition_id >= partition_count || partitions == NULL || data_matrix == NULL) return;
    
    // Get partition boundaries
    partition_t* partition = &partitions[partition_id];
    
    // Calculate partition coordinates from ID
    int partitions_x = (n + partition_size - 1) / partition_size;
    int partition_row = partition_id / partitions_x;
    int partition_col = partition_id % partitions_x;
    
    int x_start = partition_col * partition_size;
    int y_start = partition_row * partition_size;
    int x_end = min(x_start + partition_size - 1, (int)n - 1);
    int y_end = min(y_start + partition_size - 1, (int)m - 1);
    
    // Update partition boundaries
    partition->x_start = x_start;
    partition->x_end = x_end;
    partition->y_start = y_start;
    partition->y_end = y_end;
    
    // Count filled pixels using block threads
    __shared__ int block_sum;
    if (threadIdx.x == 0) {
        block_sum = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple pixels
    int thread_count = 0;
    int pixels_per_thread = ((y_end - y_start + 1) * (x_end - x_start + 1) + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < pixels_per_thread; i++) {
        int pixel_idx = threadIdx.x * pixels_per_thread + i;
        int total_pixels = (y_end - y_start + 1) * (x_end - x_start + 1);
        
        if (pixel_idx < total_pixels) {
            int local_y = pixel_idx / (x_end - x_start + 1);
            int local_x = pixel_idx % (x_end - x_start + 1);
            int global_y = y_start + local_y;
            int global_x = x_start + local_x;
            
            if (global_y < m && global_x < n) {
                if (data_matrix[global_y * n + global_x] == 1) {
                    thread_count++;
                }
            }
        }
    }
    
    // Reduce thread counts to block sum
    atomicAdd(&block_sum, thread_count);
    __syncthreads();
    
    // Calculate density
    if (threadIdx.x == 0) {
        int total_pixels = (y_end - y_start + 1) * (x_end - x_start + 1);
        partition->density = total_pixels > 0 ? (float)block_sum / (float)total_pixels : 0.0f;
    }
}

/**
 * Build connectivity graph between adjacent partitions
 * Each thread processes one partition and checks its neighbors
 * 
 * Args:
 *     partitions: Array of partitions
 *     partition_count: Number of partitions
 *     adjacency_matrix: Output connectivity matrix (partition_count x partition_count)
 *     density_threshold: Minimum density for connectivity
 *     partition_size: Size of each partition
 *     matrix_width: Width of original matrix
 */
__global__ void build_connectivity_graph(partition_t* partitions, int partition_count,
                                        int* adjacency_matrix, float density_threshold, 
                                        int partition_size, int matrix_width) {
    int partition_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (partition_id >= partition_count || partitions == NULL || adjacency_matrix == NULL) return;
    
    partition_t* current = &partitions[partition_id];
    
    // Calculate grid dimensions
    int partitions_x = (matrix_width + partition_size - 1) / partition_size;
    int partition_row = partition_id / partitions_x;
    int partition_col = partition_id % partitions_x;
    
    int connectivity_count = 0;
    
    // Check 8 neighbors (including diagonals)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // Skip self
            
            int neighbor_row = partition_row + dy;
            int neighbor_col = partition_col + dx;
            
            // Check bounds
            if (neighbor_row >= 0 && neighbor_col >= 0 && 
                neighbor_col < partitions_x && 
                neighbor_row < (partition_count / partitions_x + (partition_count % partitions_x ? 1 : 0))) {
                
                int neighbor_id = neighbor_row * partitions_x + neighbor_col;
                if (neighbor_id < partition_count) {
                    partition_t* neighbor = &partitions[neighbor_id];
                    
                    // Check if both partitions meet density threshold
                    if (current->density > density_threshold && neighbor->density > density_threshold) {
                        adjacency_matrix[partition_id * partition_count + neighbor_id] = 1;
                        connectivity_count++;
                    } else {
                        adjacency_matrix[partition_id * partition_count + neighbor_id] = 0;
                    }
                }
            }
        }
    }
    
    // Update connectivity count
    current->connectivity = connectivity_count;
}

/**
 * Update partition priorities based on density and connectivity
 * Each thread processes one partition
 * 
 * Args:
 *     partitions: Array of partitions
 *     partition_count: Number of partitions
 *     adjacency_matrix: Connectivity matrix
 */
__global__ void update_partition_priorities(partition_t* partitions, int partition_count,
                                           int* adjacency_matrix) {
    int partition_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (partition_id >= partition_count || partitions == NULL) return;
    
    partition_t* partition = &partitions[partition_id];
    
    // Validate partition boundaries before calculations
    if (partition->x_end < partition->x_start || partition->y_end < partition->y_start) {
        partition->priority = 0.0f;  // Invalid partition gets zero priority
        return;
    }
    
    // Calculate area potential (larger partitions get slight boost)
    int partition_width = partition->x_end - partition->x_start + 1;
    int partition_height = partition->y_end - partition->y_start + 1;
    
    // Safety check for reasonable partition size
    if (partition_width <= 0 || partition_height <= 0 || 
        partition_width > 128 || partition_height > 128) {
        partition->priority = 0.0f;  // Corrupted partition gets zero priority
        return;
    }
    
    float area_potential = sqrtf((float)(partition_width * partition_height)) / 32.0f;  // Normalize to ~32x32
    
    // Validate density before use
    float density = partition->density;
    if (density < 0.0f) density = 0.0f;
    if (density > 1.0f) density = 1.0f;
    
    // Validate connectivity before use
    int connectivity = partition->connectivity;
    if (connectivity < 0) connectivity = 0;
    if (connectivity > 8) connectivity = 8;  // Max 8 neighbors
    
    // Priority formula: density × connectivity × area_potential
    // Add small epsilon to avoid zero priorities
    partition->priority = (density + 0.01f) * (connectivity + 1.0f) * area_potential;
    
    // Clamp priority to reasonable range
    if (partition->priority < 0.0f) partition->priority = 0.01f;
    if (partition->priority > 1000.0f) partition->priority = 1000.0f;  // Cap at reasonable max
}

/**
 * Update affected partitions after rectangle removal
 * Only recalculates density for partitions that intersect with removed rectangle
 * 
 * Args:
 *     x1, x2, y1, y2: Rectangle boundaries that was removed
 *     partitions: Array of partitions
 *     partition_count: Number of partitions
 *     data_matrix: Updated matrix after rectangle removal
 *     m: Matrix height
 *     n: Matrix width
 *     partition_size: Size of each partition
 */
__global__ void update_affected_partitions(int x1, int x2, int y1, int y2,
                                          partition_t* partitions, int partition_count,
                                          int* data_matrix, long m, long n, int partition_size) {
    int partition_id = blockIdx.x;
    if (partition_id >= partition_count || partitions == NULL || data_matrix == NULL) return;
    
    // Additional safety checks for matrix dimensions
    if (m <= 0 || n <= 0 || partition_size <= 0) return;
    if (x1 < 0 || x2 >= n || y1 < 0 || y2 >= m || x2 < x1 || y2 < y1) return;
    
    partition_t* partition = &partitions[partition_id];
    
    // Validate partition boundaries before use
    if (partition->x_start < 0 || partition->x_end >= n || 
        partition->y_start < 0 || partition->y_end >= m ||
        partition->x_end < partition->x_start || partition->y_end < partition->y_start) {
        // Partition boundaries invalid, skip update
        return;
    }
    
    // Check if partition intersects with removed rectangle (now with validated boundaries)
    bool intersects = !(x2 < partition->x_start || x1 > partition->x_end || 
                       y2 < partition->y_start || y1 > partition->y_end);
    
    if (!intersects) return;
    
    // Recalculate density for affected partition with robust bounds checking
    __shared__ int block_sum;
    if (threadIdx.x == 0) {
        block_sum = 0;
    }
    __syncthreads();
    
    int thread_count = 0;
    int partition_width = partition->x_end - partition->x_start + 1;
    int partition_height = partition->y_end - partition->y_start + 1;
    int total_pixels = partition_width * partition_height;
    
    // Safety check for partition size
    if (total_pixels <= 0 || total_pixels > partition_size * partition_size * 4) {
        return; // Skip if partition seems corrupted
    }
    
    int pixels_per_thread = (total_pixels + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < pixels_per_thread; i++) {
        int pixel_idx = threadIdx.x * pixels_per_thread + i;
        
        if (pixel_idx < total_pixels) {
            int local_y = pixel_idx / partition_width;
            int local_x = pixel_idx % partition_width;
            int global_y = partition->y_start + local_y;
            int global_x = partition->x_start + local_x;
            
            // Double-check bounds before memory access
            if (global_y >= 0 && global_y < m && global_x >= 0 && global_x < n) {
                long matrix_idx = global_y * n + global_x;
                if (matrix_idx >= 0 && matrix_idx < m * n) {  // Final bounds check
                    if (data_matrix[matrix_idx] == 1) {
                        thread_count++;
                    }
                }
            }
        }
    }
    
    atomicAdd(&block_sum, thread_count);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        partition->density = total_pixels > 0 ? (float)block_sum / (float)total_pixels : 0.0f;
        // Clamp density to valid range
        if (partition->density < 0.0f) partition->density = 0.0f;
        if (partition->density > 1.0f) partition->density = 1.0f;
    }
}

#endif // PARTITION_KERNELS_H 