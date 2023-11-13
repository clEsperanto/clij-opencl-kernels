__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void range(
    IMAGE_dst_TYPE  dst,
    IMAGE_src_TYPE  src,
    const int       start_x,
    const int       start_y,    
    const int       start_z,
    const int       step_x,
    const int       step_y,
    const int       step_z
) 
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  const int dz = get_global_id(2);

  const int sx = get_global_id(0) * step_x + start_x;
  const int sy = get_global_id(1) * step_y + start_y;
  const int sz = get_global_id(2) * step_z + start_z;

  const float out = READ_IMAGE(src, sampler, POS_src_INSTANCE(sx, sy, sz, 0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(dx, dy, dz, 0), CONVERT_dst_PIXEL_TYPE(out));
}
