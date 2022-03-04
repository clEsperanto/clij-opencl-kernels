__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void multiply_matrix(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float sum = 0;
  for (int i = 0; i < GET_IMAGE_WIDTH(src0); ++i) {
      sum += READ_IMAGE(src0, sampler, POS_src0_INSTANCE(i,y,0,0)).x * READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x,i,0,0)).x;
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,0,0), CONVERT_dst_PIXEL_TYPE(sum));
}
