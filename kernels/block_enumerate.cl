__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void block_enumerate(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst,
    const int        index
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  float sum = 0;
  for (int sx = 0; sx < x; ++sx) {
    sum += (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(sx,y,z,0)).x;
  }

  for (int dx = 0; dx < index; ++dx) {
    float value = (float) READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x * index + dx,y,z,0)).x;
    if (value != 0) {
      sum += value;
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x * index + dx,y,z,0), CONVERT_dst_PIXEL_TYPE(sum));
    } else {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x * index + dx,y,z,0), 0);
    }
  }
}
