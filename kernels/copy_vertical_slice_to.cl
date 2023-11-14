__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void copy_vertical_slice_to(
    IMAGE_src_TYPE  src, 
    IMAGE_dst_TYPE  dst, 
    const int       index
) 
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  const POS_src_TYPE pos_src = POS_src_INSTANCE(y, z, 0, 0);
  const POS_dst_TYPE pos_dst = POS_dst_INSTANCE(index, y, z, 0);

  const float value = (float) READ_IMAGE(src, sampler, pos_src).x;
  WRITE_IMAGE(dst, pos_dst, CONVERT_dst_PIXEL_TYPE(value));
}