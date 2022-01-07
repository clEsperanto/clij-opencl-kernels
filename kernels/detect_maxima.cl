__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_maxima(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{ 
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  
  int4 r = (int4){0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { r.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { r.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { r.z = 1; }

  POS_src_TYPE localMaxPos = POS_src_INSTANCE(i,j,k,0);
  POS_src_TYPE localPos = POS_src_INSTANCE(i,j,k,0);  
  IMAGE_src_PIXEL_TYPE localMax = READ_IMAGE(src, sampler, localPos).x - 1;
  for (int x = -r.x; x <= r.x; ++x) {
      for (int y = -r.y; y <= r.y; ++y) {
          for (int z = -r.z; z <= r.z; ++z) {
              POS_src_TYPE localPos = localMaxPos + POS_src_INSTANCE(x,y,z,0);
              const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, localPos).x;
              if (value > localMax) {
                  localMax = value;
                  localMaxPos = localPos;
              }
          }
      }
  }
  IMAGE_dst_PIXEL_TYPE result = 0;
  if (all(localPos == localMaxPos)) {
      result = 1;
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(i,j,k,0), result);
}
