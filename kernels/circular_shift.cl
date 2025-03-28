__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void circular_shift(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       index_1,
    const int       index_2,
    const int       index_3
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);

  int x_shifted = x;
  int y_shifted = y;
  int z_shifted = z;
  
  if (width > 1 && index_1 != 0) {
    x_shifted = (x + index_1);
    if (x_shifted >= width) {
      x_shifted -= width;
    }
    if (x_shifted < 0) {
      x_shifted += width;
    }
  }
  if (height > 1 && index_2 != 0) {
    y_shifted = (y + index_2);
    if (y_shifted >= height) {
      y_shifted -= height;
    }
    if (y_shifted < 0) {
      y_shifted += height;
    }
  }
  if (depth > 1 && index_3 != 0) {
    z_shifted = (z + index_3);
    if (z_shifted >= depth) {
      z_shifted -= depth;
    }
    if (z_shifted < 0) {
      z_shifted += depth;
    }
  }

  const POS_src_TYPE coord_in = POS_src_INSTANCE(x, y, z, 0);
  const POS_dst_TYPE coord_out = POS_dst_INSTANCE(x_shifted, y_shifted, z_shifted, 0);

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, coord_in).x;
  WRITE_IMAGE(dst, coord_out, CONVERT_dst_PIXEL_TYPE(value));
}
