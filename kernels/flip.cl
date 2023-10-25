__constant sampler_t sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void flip(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       index0,
    const int       index1,
    const int       index2
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int w = GET_IMAGE_WIDTH(src);
  const int h = GET_IMAGE_HEIGHT(src);
  const int d = GET_IMAGE_DEPTH(src);

  const int dx = index0 ? (w - 1 - x) : x;
  const int dy = index1 ? (h - 1 - y) : y;
  const int dz = index2 ? (d - 1 - z) : z;

  const IMAGE_src_PIXEL_TYPE value = READ_src_IMAGE(src, sampler, POS_src_INSTANCE(dx,dy,dz,0)).x;
  WRITE_dst_IMAGE(dst, POS_src_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
