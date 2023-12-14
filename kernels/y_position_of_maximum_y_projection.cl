__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void y_position_of_maximum_y_projection (
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{

  const int x = get_global_id(0);
  const int z = get_global_id(1);
  float max = 0;
  int max_pos = 0;
  for(int y = 0; y < GET_IMAGE_HEIGHT(src); y++)
  {
    float value = READ_IMAGE(src,sampler,POS_src_INSTANCE(x,y,z,0)).x;
    if (value > max || y == 0) {
      max = value;
      max_pos = y;
    }
  }
  WRITE_IMAGE(dst,POS_dst_INSTANCE(x,z,0,0), CONVERT_dst_PIXEL_TYPE(max_pos));
}