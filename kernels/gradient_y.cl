__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gradient_y(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const float valueA = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(i,j-1,k,0)).x;
  const float valueB = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(i,j+1,k,0)).x;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(i,j,k,0), CONVERT_dst_PIXEL_TYPE(valueB - valueA));
}
