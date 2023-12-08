const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Convert 3D coordinate to linear index
inline int xyz_to_linear(int4 coord, int width, int height, int depth) {
    return (coord.z * width * height) + (coord.y * width) + coord.x;
}

// Convert linear index to 3D coordinate
inline int4 linear_to_xyz(int index, int width, int height, int depth) {
    int4 res = (int4)(0, 0, 0, 0);
    res.z = index / (width * height);
    index -= res.z * width * height;
    res.y = index / width;
    res.x = index % width;
    return res;
}

__kernel void reshape(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE   dst
)
{
  const int dw = GET_IMAGE_WIDTH(dst);
  const int dh = GET_IMAGE_HEIGHT(dst);
  const int dd = GET_IMAGE_DEPTH(dst);

  const int sw = GET_IMAGE_WIDTH(src);
  const int sh = GET_IMAGE_HEIGHT(src);
  const int sd = GET_IMAGE_DEPTH(src);

  const int4 coord_dst = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
  const int linear_index = xyz_to_linear(coord_dst, dw, dh, dd);
  const int4 coord_src = linear_to_xyz(linear_index, sw, sh, sd);

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(coord_src.x,coord_src.y,coord_src.z,0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(coord_src.x,coord_src.y,coord_src.z,0), CONVERT_dst_PIXEL_TYPE(value));
}
