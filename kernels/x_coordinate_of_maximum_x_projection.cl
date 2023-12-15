__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void x_coordinate_of_maximum_x_projection (
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{

  const int z = get_global_id(0);
  const int y = get_global_id(1);
  float max = 0;
  int max_pos = 0;
  for(int x = 0; x < GET_IMAGE_WIDTH(src); x++)
  {
    float value = READ_IMAGE(src,sampler,POS_src_INSTANCE(x,y,z,0)).x;
    if (value > max || x == 0) {
      max = value;
      max_pos = x;
    }
  }
  WRITE_IMAGE(dst,POS_dst_INSTANCE(z,y,0,0), CONVERT_dst_PIXEL_TYPE(max_pos));
}