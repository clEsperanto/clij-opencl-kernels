__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// TILE_SIZE SELECTION RULE:
// The TILE_SIZE should be chosen based on matrix dimensions to optimize performance
// without wasting resources on oversized work groups.
// The TILE_SIZE should be chosen based on hardware limits and capabilities.
//
// Recommended rule:
//   TILE_SIZE = min(32, nearest_power_of_2(sqrt(min(src0_width, src1_height))))
//
// Examples:
//   - src0: 2x3, src1: 4x2  -> min(3,2)=2   -> sqrt(2)≈1.4  -> TILE_SIZE=2 (or 1)
//   - src0: 32x64, src1: 128x32 -> min(64,32)=32 -> sqrt(32)≈5.6 -> TILE_SIZE=8
//   - src0: 256x512, src1: 1024x256 -> min(512,256)=256 -> sqrt(256)=16 -> TILE_SIZE=16
//   - src0: 1024x2048, src1: 4096x1024 -> min(2048,1024)=1024 -> sqrt(1024)≈32 -> TILE_SIZE=32
//
// This ensures:
// - Work group dimensions match problem size (no oversized groups for small matrices)
// - Efficient local memory usage
// - Good GPU utilization for all matrix sizes
// - Proper synchronization without divergence

__kernel void multiply_matrix(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  #if TILE_SIZE == 1
    // For TILE_SIZE of 1, local IDs are always 0
    const int local_x = 0;
    const int local_y = 0;
  #else
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
  #endif
  const int src0_width = GET_IMAGE_WIDTH(src0);
  const int src1_width = GET_IMAGE_WIDTH(src1);
  const int src0_height = GET_IMAGE_HEIGHT(src0);
  const int src1_height = GET_IMAGE_HEIGHT(src1);

  __local float tile_src0[TILE_SIZE][TILE_SIZE];
  __local float tile_src1[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  
  // Check if this work item is within valid output bounds
  const bool is_valid_output = (x < src1_width && y < src0_height);

  // Process matrix in tiles
  const int num_tiles = (src0_width + TILE_SIZE - 1) / TILE_SIZE;
  for (int tile = 0; tile < num_tiles; ++tile) {
      // Load tiles into local memory
      const int tile_col = tile * TILE_SIZE + local_x;
      const int tile_row = tile * TILE_SIZE + local_y;

      // Load src0: columns from src0 for current row
      // All work items participate in loading to avoid divergence
      if (tile_col < src0_width && y < src0_height) {
            tile_src0[local_y][local_x] = READ_IMAGE(src0, sampler, POS_src0_INSTANCE(tile_col, y, 0, 0)).x;
      } else {
            tile_src0[local_y][local_x] = 0.0f;
      }
      
      // Load src1: rows from src1 for current column
      // All work items participate in loading to avoid divergence
      if (tile_row < src1_height && x < src1_width) {
            tile_src1[local_y][local_x] = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x, tile_row, 0, 0)).x;
      } else {
            tile_src1[local_y][local_x] = 0.0f;
      }

      // Synchronize to ensure all work items have finished loading tiles
      barrier(CLK_LOCAL_MEM_FENCE);

      // Compute partial dot product only for valid output positions
      if (is_valid_output) {
#if TILE_SIZE == 1
          // Simple scalar accumulation for TILE_SIZE=1
          sum += tile_src0[0][0] * tile_src1[0][0];
#else
          // Loop through all elements in the tile
          for (int i = 0; i < TILE_SIZE; ++i) {
              sum += tile_src0[local_y][i] * tile_src1[i][local_x];
          }
#endif
      }

      // Synchronize before next tile iteration
      // All work items must reach this barrier regardless of validity
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  // Write output only for valid positions
  if (is_valid_output) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, 0, 0), CONVERT_dst_PIXEL_TYPE(sum));
  }
}