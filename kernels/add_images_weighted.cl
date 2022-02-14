__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void add_images_weighted(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst,
    const float      scalar0,
    const float      scalar1
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const float value0 = scalar0 * (float) READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x,y,z,0)).x;
  const float value1 = scalar1 * (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x,y,z,0)).x;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value0 + value1));
}
