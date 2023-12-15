__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void z_position_of_maximum_z_projection (
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  float max = 0;
  int max_pos = 0;
  for(int z = 0; z < GET_IMAGE_DEPTH(src); z++)
  {
    float value = READ_IMAGE(src,sampler,POS_src_INSTANCE(x,y,z,0)).x;
    if (value > max || z == 0) {
      max = value;
      max_pos = z;
    }
  }
  WRITE_IMAGE(dst,POS_dst_INSTANCE(x,y,0,0), CONVERT_dst_PIXEL_TYPE(max_pos));
}
