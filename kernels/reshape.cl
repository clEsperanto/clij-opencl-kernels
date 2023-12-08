const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Convert 3D coordinate to linear index
inline void xyz_to_linear(int x, int y, int z, int width, int height, int depth, int &index) {
    index = (z * width * height) + (y * width) + x;
}

// Convert linear index to 3D coordinate
inline void linear_to_xyz(int index, int width, int height, int depth, int &sx, int &sy, int &sz) {
    sz = index / (width * height);
    index -= sz * width * height;
    sy = index / width;
    sx = index % width;
}

__kernel void reshape(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE   dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int dw = GET_IMAGE_WIDTH(dst);
  const int dh = GET_IMAGE_HEIGHT(dst);
  const int dd = GET_IMAGE_DEPTH(dst);

  const int sw = GET_IMAGE_WIDTH(src);
  const int sh = GET_IMAGE_HEIGHT(src);
  const int sd = GET_IMAGE_DEPTH(src);

  int sx = 0;
  int sy = 0;
  int sz = 0;

  int linear_index = 0;
  xyz_to_linear(x, y, z, dw, dh, dd, linear_index);
  linear_to_xyz(linear_index, sw, sh, sd, sx, sy, sz);

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(sx,sy,sz,0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
