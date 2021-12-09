__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void downsample(
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst,
    const float    scalar0,
    const float    scalar1,
    const float    scalar2
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int dx = scalar0 * x;
  const int dy = scalar1 * y;
  const int dz = scalar2 * z;

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(dx,dy,dz,0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}