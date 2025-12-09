__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void multiply_matrix(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);

  const int src0_width = GET_IMAGE_WIDTH(src0);
  const int src1_width = GET_IMAGE_WIDTH(src1);
  const int src0_height = GET_IMAGE_HEIGHT(src0);
  const int src1_height = GET_IMAGE_HEIGHT(src1);

  __local float tile_src0[TILE_SIZE][TILE_SIZE];
  __local float tile_src1[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  // Process matrix in tiles
  for (int tile = 0; tile < (src0_width + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
      // Load tiles into local memory
#if TILE_SIZE == 1
      const int tile_col = tile;
      const int tile_row = tile;
#else
      const int tile_col = tile * TILE_SIZE + local_x;
      const int tile_row = tile * TILE_SIZE + local_y;
#endif

      if (tile_col < src0_width && y < src0_height) {
          tile_src0[local_y][local_x] = READ_IMAGE(src0, sampler, POS_src0_INSTANCE(tile_col, y, 0, 0)).x;
      } else {
          tile_src0[local_y][local_x] = 0;
      }

      if (tile_row < src1_height && x < src1_width) {
          tile_src1[local_y][local_x] = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x, tile_row, 0, 0)).x;
      } else {
          tile_src1[local_y][local_x] = 0;
      }

      // Synchronize to ensure all work items have finished loading tiles
      barrier(CLK_LOCAL_MEM_FENCE);

      // Compute partial dot product
#if TILE_SIZE == 1
      sum += tile_src0[0][0] * tile_src1[0][0];
#else
      for (int i = 0; i < TILE_SIZE; i += 4) {
          sum += tile_src0[local_y][i]     * tile_src1[i][local_x];
          sum += tile_src0[local_y][i + 1] * tile_src1[i + 1][local_x];
          sum += tile_src0[local_y][i + 2] * tile_src1[i + 2][local_x];
          sum += tile_src0[local_y][i + 3] * tile_src1[i + 3][local_x];
      }
#endif

      // not needed (?)
      // barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, 0, 0), CONVERT_dst_PIXEL_TYPE(sum));
}