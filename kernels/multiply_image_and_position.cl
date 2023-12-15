__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void multiply_image_and_position(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       index
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int coord = 0;
  if      (index == 0) {coord = x;}
  else if (index == 1) {coord = y;}
  else if (index == 2) {coord = z;}

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value * coord));
}
