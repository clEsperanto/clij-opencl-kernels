
__kernel void set_image_borders(
    IMAGE_src_TYPE  src,
    const float     scalar
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);

  const int is_border = (width > 1 && (x == 0 || x == width - 1)) ||
                        (height > 1 && (y == 0 || y == height - 1)) ||
                        (depth > 1 && (z == 0 || z == depth - 1));
  
  if (is_border) {
    WRITE_IMAGE(src, POS_src_INSTANCE(x,y,z,0), CONVERT_src_PIXEL_TYPE(scalar));
  }
}
