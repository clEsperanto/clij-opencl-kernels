__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void minimum_of_masked_pixels_reduction(
    IMAGE_src_TYPE       src,
    IMAGE_mask_TYPE      mask,
    IMAGE_dst_src_TYPE   dst_src,
    IMAGE_dst_mask_TYPE  dst_mask
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int depth = GET_IMAGE_DEPTH(src);

  float minimum = FLT_MAX;
  float value = FLT_MAX;
  float mask_value = 0;
  float binary = 0;
  for(int z = 0; z < depth; ++z)
  {
    binary = (float) READ_IMAGE(mask, sampler, POS_mask_INSTANCE(x, y, z, 0)).x;
    if (binary != 0) 
    {
        mask_value = 1;
        value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
        minimum = (value < minimum)? value : minimum;
    }
  }
  WRITE_IMAGE(dst_src, POS_dst_src_INSTANCE(x, y, 0, 0), CONVERT_dst_src_PIXEL_TYPE(minimum));
  WRITE_IMAGE(dst_mask, POS_dst_mask_INSTANCE(x, y, 0, 0), CONVERT_dst_mask_PIXEL_TYPE(mask_value));
}