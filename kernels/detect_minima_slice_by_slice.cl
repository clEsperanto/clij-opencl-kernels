__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_minima_slice_by_slice(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int radius = 1;
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);
  POS_src_TYPE localMinPos = POS_src_INSTANCE(x,y,z,0);
  IMAGE_src_PIXEL_TYPE localMin = READ_IMAGE(src, sampler, pos).x - 1;
  for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
          const POS_src_TYPE localPos = pos + POS_src_TYPE(dx,dy,0,0);
          const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, localPos).x;
          if (value < localMin) {
              localMin = value;
              localMinPos = localPos;
          }
      }
  }
  
  if (x == localMinPos.x && y == localMinPos.y) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), 1);
  } else {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), 0);
  }
}