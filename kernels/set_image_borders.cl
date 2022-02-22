
__kernel void set_image_borders(
    IMAGE_dst_TYPE  dst,
    const float     scalar
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  if (GET_IMAGE_WIDTH(dst) > 1 && (x == 0 || x == GET_IMAGE_WIDTH(dst) - 1))
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
  } 
  else if (GET_IMAGE_HEIGHT(dst) > 1 && (y == 0 || y == GET_IMAGE_HEIGHT(dst) - 1))
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
  } 
  else if (GET_IMAGE_DEPTH(dst) > 1 && (z == 0 || z == GET_IMAGE_DEPTH(dst) - 1))
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
  } 
}
