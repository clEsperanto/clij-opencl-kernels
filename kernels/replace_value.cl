__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void replace_value(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const float     scalar0,
    const float     scalar1
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const IMAGE_dst_PIXEL_TYPE target_value = CONVERT_src_PIXEL_TYPE(scalar0);
  const IMAGE_dst_PIXEL_TYPE replacement_value = CONVERT_src_PIXEL_TYPE(scalar1);
  
  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  int condition = (value == target_value);
  const IMAGE_dst_PIXEL_TYPE output = (condition * replacement_value) + ((1 - condition) * CONVERT_dst_PIXEL_TYPE(value));
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), output);
}
