__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void transpose_xy(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE spos = POS_src_INSTANCE(y, x, z, 0);
  const POS_dst_TYPE dpos = POS_dst_INSTANCE(x, y, z, 0);

  float value = READ_src_IMAGE(src, sampler, spos).x;

  WRITE_IMAGE(dst, dpos, CONVERT_dst_PIXEL_TYPE(value));
}
