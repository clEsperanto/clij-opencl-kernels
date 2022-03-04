__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void reslice_right( 
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst,
) 
{
  const int x = get_global_id(2);
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  const int dz = get_global_size(2) - get_global_id(2) - 1;

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(dx,dy,dz,0), CONVERT_dst_PIXEL_TYPE(value));
}
