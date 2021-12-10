__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void set_row(
    IMAGE_dst_TYPE  dst,
    const int       index,
    const float     scalar
)
{
  const int x = get_global_id(0);
  const int y = index;
  const int z = get_global_id(2);

  WRITE_IMAGE(dst, POS_dst_INSTANT(x,y,z,0), CONVERT_dst_PIXEL_TYPE(scalar));
}