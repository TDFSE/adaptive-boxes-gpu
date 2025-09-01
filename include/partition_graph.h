#ifndef PARTITION_GRAPH_H
#define PARTITION_GRAPH_H

#include <cuda_runtime.h>

/**
 * Partition structure representing a spatial tile in the 2D matrix
 * Used for spatial partitioning to guide rectangle exploration
 */
struct partition_t {
    int x_start, x_end;    // Tile boundaries (inclusive)
    int y_start, y_end;
    float density;         // Ratio of filled pixels (0.0-1.0)
    int connectivity;      // Number of connected neighbors
    float priority;        // Search priority score
};

/**
 * Partition graph structure for spatial decomposition
 * Manages partitions and their connectivity relationships
 */
struct partition_graph_t {
    partition_t* partitions;   // Device array of partitions
    int partition_count;       // Total number of partitions  
    int* adjacency_matrix;     // Sparse connectivity matrix (partition_count x partition_count)
    int* priority_queue;       // Sorted partition indices by priority
    int partition_size;        // Size of each partition (e.g., 32x32)
};

/**
 * Helper functions for partition management
 */

/**
 * Calculate number of partitions needed for given matrix dimensions
 * 
 * Args:
 *     m: Matrix height
 *     n: Matrix width
 *     partition_size: Size of each partition tile
 * 
 * Returns:
 *     Total number of partitions needed
 */
__host__ __device__ inline int calculate_partition_count(int m, int n, int partition_size) {
    int partitions_y = (m + partition_size - 1) / partition_size;  // Ceiling division
    int partitions_x = (n + partition_size - 1) / partition_size;
    return partitions_x * partitions_y;
}

/**
 * Get partition index for given matrix coordinates
 * 
 * Args:
 *     row: Matrix row
 *     col: Matrix column
 *     n: Matrix width
 *     partition_size: Size of each partition tile
 * 
 * Returns:
 *     Partition index
 */
__host__ __device__ inline int get_partition_index(int row, int col, int n, int partition_size) {
    int partition_row = row / partition_size;
    int partition_col = col / partition_size;
    int partitions_x = (n + partition_size - 1) / partition_size;
    return partition_row * partitions_x + partition_col;
}

/**
 * Get priority-guided partition for this thread (avoids clustering)
 * Each thread gets a different high-priority partition using thread ID
 * 
 * Args:
 *     partitions: Array of partitions
 *     partition_count: Number of partitions
 *     thread_id: Unique thread identifier for distribution
 * 
 * Returns:
 *     Index of a high-priority partition for this thread
 */
__device__ inline int get_priority_guided_partition(partition_t* partitions, int partition_count, int thread_id) {
    if (partitions == NULL || partition_count <= 0) return 0;
    
    // Fast sampling approach: avoid expensive linear search
    // Sample from a smaller subset of partitions and pick the best among them
    const int sample_size = min(8, partition_count);  // Sample at most 8 partitions
    
    int best_idx = 0;
    float best_priority = -1.0f;
    
    for (int i = 0; i < sample_size; i++) {
        // Use thread_id and iteration to distribute sampling across partition space
        int sample_idx = (thread_id + i * 1337) % partition_count;  // Pseudo-random distribution
        
        float priority = partitions[sample_idx].priority;
        if (priority > best_priority) {
            best_priority = priority;
            best_idx = sample_idx;
        }
    }
    

    return best_idx;
}

#endif // PARTITION_GRAPH_H