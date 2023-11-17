__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void generate_distance_matrix(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
) 
{
  const int x = get_global_id(0);

  const int n_dimensions = GET_IMAGE_HEIGHT(src0);
  const int n_points = GET_IMAGE_WIDTH(src1);

  float positions[10];
  for (int i = 0; i < n_dimensions; i++) {
      positions[i] = READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x, i, 0, 0)).x;
  }

  for (int j = 0; j < n_points; j++) {
      float sum = 0;
      for (int i = 0; i < n_dimensions; i ++) {
          const float value = positions[i] - (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(j, i, 0, 0)).x
          sum = sum + (value * value);
      }
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x+1, j+1, 0, 0), CONVERT_dst_PIXEL_TYPE(sqrt(sum)));
  }
}