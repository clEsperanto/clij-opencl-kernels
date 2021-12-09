__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void set_nonzero_pixels_to_pixelindex(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       cst
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const int w = GET_IMAGE_WIDTH(src);
  const int h = GET_IMAGE_HEIGHT(src);
  const int d = GET_IMAGE_DEPTH(src);

  IMAGE_dst_PIXEL_TYPE pixelindex = CONVERT_dst_PIXEL_TYPE(i * h * d + j * d + k + cst);
  IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(i,j,k,0)).x;
  if (value == 0) {
      pixelindex = 0;
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(i,j,k,0), pixelindex);
}