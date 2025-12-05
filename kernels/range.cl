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

  const int4 src_coords = (int4)(dx * step_x + start_x, dy * step_y + start_y, dz * step_z + start_z, 0);
  const float out = READ_IMAGE(src, sampler, POS_src_INSTANCE(src_coords.x, src_coords.y, src_coords.z, 0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(dx, dy, dz, 0), CONVERT_dst_PIXEL_TYPE(out));
}
