__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void not_equal_constant(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const float     scalar,
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  IMAGE_dst_PIXEL_TYPE value = 0;
  const IMAGE_src_PIXEL_TYPE input = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  if (input != scalar) {
    value = 1;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), value);
}