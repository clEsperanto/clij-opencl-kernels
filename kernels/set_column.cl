__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void set_column(
    IMAGE_dst_TYPE  dst,
    const int       index,
    const float     scalar
)
{
  const int x = index;
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(scalar));
}
