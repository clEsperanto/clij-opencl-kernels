__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void variance_box(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       scalar0,
    const int       scalar1,
    const int       scalar2
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE coord = POS_src_INSTANCE(x,y,z,0);
  const int4 radius = (int4){(scalar0-1)/2, (scalar1-1)/2, (scalar2-1)/2, 0};
  
  int count = 0;
  float sum = 0;
      for (int dz = -radius.z; dz <= radius.z; ++dz) {
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
  for (int dx = -radius.x; dx <= radius.x; ++dx) {
          const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz, 0);
          sum = sum + (float) READ_IMAGE(src, sampler, coord + pos).x;
          count = count + 1;
      }
    }
  }
  const float mean_intensity = sum / count;
  sum = 0;
  count = 0;
      for (int dz = -radius.z; dz <= radius.z; ++dz) {
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
  for (int dx = -radius.x; dx <= radius.x; ++dx) {
          const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz, 0);
          const float value = (float) READ_IMAGE(src, sampler, coord + pos).x;
          sum = sum + pow(value - mean_intensity, 2);
          count = count + 1;
      }
    }
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(sum / count));
}