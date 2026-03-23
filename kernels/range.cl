__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void range(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       start_x,
    const int       step_x,
    const int       start_y,    
    const int       step_y,
    const int       start_z,
    const int       step_z
) 
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  const int dz = get_global_id(2);

  if (dx >= GET_WIDTH(dst) || dy >= GET_HEIGHT(dst) || dz >= GET_DEPTH(dst))
    return;

  const int sx = start_x + dx * step_x;
  const int sy = start_y + dy * step_y;
  const int sz = start_z + dz * step_z;

  if (sx < 0 || sx >= GET_WIDTH(src) || sy < 0 || sy >= GET_HEIGHT(src) || sz < 0 || sz >= GET_DEPTH(src))
    return;

  const float out = READ_IMAGE(src, sampler, POS_src_INSTANCE(sx, sy, sz, 0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(dx, dy, dz, 0), CONVERT_dst_PIXEL_TYPE(out));
}
