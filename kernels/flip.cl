__constant sampler_t sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void flip(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       scalar0,
    const int       scalar1,
    const int       scalar2
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int dx = scalar0 ? (get_global_size(0) - 1 - x) : x;
  const int dy = scalar1 ? (get_global_size(1) - 1 - y) : y;
  const int dz = scalar2 ? (get_global_size(2) - 1 - z) : z;

  const IMAGE_src_PIXEL_TYPE value = READ_src_IMAGE(src, sampler, POS_src_INSTANCE(dx,dy,dz,0)).x;
  WRITE_dst_IMAGE(dst, POS_src_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}