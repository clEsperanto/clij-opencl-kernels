__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void mask(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  IMAGE_src0_PIXEL_TYPE value = 0;
  const IMAGE_src1_PIXEL_TYPE mask = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x,y,z,0)).x;
  if (mask != 0) {
    value = READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x,y,z,0)).x;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
