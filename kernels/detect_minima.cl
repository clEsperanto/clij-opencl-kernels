__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_minima(
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

  IMAGE_src_PIXEL_TYPE localMin = READ_IMAGE(src, sampler, POS_src_INSTANCE(i,j,k,0)).x - 1;
  int4 localMinPos = (int4){i,j,k,0};
  for (int x = -r.x; x <= r.x; ++x) {
      for (int y = -r.y; y <= r.y; ++y) {
          for (int z = -r.z; z <= r.z; ++z) {
              int4 localPos = localMinPos + (int4){x,y,z,0};
              const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(localPos.x,localPos.y,localPos.z,0)).x;
              if (value < localMin) {
                  localMin = value;
                  localMinPos = localPos;
              }
          }
      }
  }

  IMAGE_dst_PIXEL_TYPE result = 0;
  if (localMinPos.x == i && localMinPos.y == j && localMinPos.z == k) {
      result = 1;
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(i,j,k,0), result);
}
