
__kernel void set_image_borders(
    IMAGE_dst_TYPE  dst,
    const float     scalar
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int width = GET_IMAGE_WIDTH(dst);
  const int height = GET_IMAGE_HEIGHT(dst);
  const int depth = GET_IMAGE_DEPTH(dst);

  if (width > 1 && (x == 0 || x == width - 1))
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(scalar));
  } 
  else if (height > 1 && (y == 0 || y == height - 1))
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(scalar));
  } 
  else if (depth > 1 && (z == 0 || z == depth - 1))
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(scalar));
  } 
}
