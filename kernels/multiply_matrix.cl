__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/// Naive matrix multiplication kernel
// __kernel void multiply_matrix(
//     IMAGE_src0_TYPE  src0,
//     IMAGE_src1_TYPE  src1,
//     IMAGE_dst_TYPE   dst
// ) 
// {
//   const int x = get_global_id(0);
//   const int y = get_global_id(1);
//   float sum = 0;
//   for (int i = 0; i < GET_IMAGE_WIDTH(src0); ++i) {
//       sum += READ_IMAGE(src0, sampler, POS_src0_INSTANCE(i,y,0,0)).x * READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x,i,0,0)).x;
//   }
//   WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,0,0), CONVERT_dst_PIXEL_TYPE(sum));
// }

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
  
  __local float tile_src0[TILE_SIZE][TILE_SIZE];
  __local float tile_src1[TILE_SIZE][TILE_SIZE];
  
  float sum = 0;
  
  // Process matrix in tiles
  for (int tile = 0; tile < (src0_width + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
      // Load tiles into local memory
      int tile_col = tile * TILE_SIZE + local_x;
      int tile_row = tile * TILE_SIZE + local_y;
      
      if (tile_col < src0_width && y < GET_IMAGE_HEIGHT(src0)) {
          tile_src0[local_y][local_x] = READ_IMAGE(src0, sampler, POS_src0_INSTANCE(tile_col, y, 0, 0)).x;
      } else {
          tile_src0[local_y][local_x] = 0;
      }
      
      if (tile_row < GET_IMAGE_HEIGHT(src1) && x < GET_IMAGE_WIDTH(src1)) {
          tile_src1[local_y][local_x] = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x, tile_row, 0, 0)).x;
      } else {
          tile_src1[local_y][local_x] = 0;
      }
      
      barrier(CLK_LOCAL_MEM_FENCE);
      
      // Compute partial dot product
      for (int i = 0; i < TILE_SIZE; ++i) {
          sum += tile_src0[local_y][i] * tile_src1[i][local_x];
      }
      
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, 0, 0), CONVERT_dst_PIXEL_TYPE(sum));
}