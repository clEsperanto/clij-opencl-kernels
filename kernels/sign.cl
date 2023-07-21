__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sign(
    IMAGE_dst_TYPE  dst,
    IMAGE_src_TYPE  src
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  float value = (float) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;

  if (isnan(value)) {
    // keep nan
  } else if (value < 0) {
    value = -1;
  } else if (value > 0) {
    value = 1;
  } else {
    value = 0;
  }
  WRITE_dst_IMAGE(dst, POS_dst_INSTANCE(x, y, z,0), CONVERT_dst_PIXEL_TYPE(value));
}
