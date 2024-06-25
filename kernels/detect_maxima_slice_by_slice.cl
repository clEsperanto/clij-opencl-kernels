__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_maxima_slice_by_slice(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int radius = 1;
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);
  POS_src_TYPE localMaxPos = POS_src_INSTANCE(x,y,z,0);
  float localMax = (float) READ_IMAGE(src, sampler, pos).x - 1;
  for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
          const POS_src_TYPE localPos = pos + POS_src_TYPE(dx,dy,0,0);
          if( all(localPos >= 0) && any(localPos != pos) ) {
            const float value = READ_IMAGE(src, sampler, localPos).x;
            if (value > localMax) {
                localMax = value;
                localMaxPos = localPos;
            }
          }
      }
  }
  
  if (x == localMaxPos.x && y == localMaxPos.y) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), 1);
  } else {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), 0);
  }
}
