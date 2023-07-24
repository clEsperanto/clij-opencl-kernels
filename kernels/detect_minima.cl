__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_minima(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{ 
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int4 r = (int4){0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { r.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { r.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { r.z = 1; }

  const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
  POS_src_TYPE localMinPos = pos;
  float localMin = (float) READ_IMAGE(src, sampler, pos).x -1;
  for (int rx = -r.x; rx <= r.x; ++rx) {
      for (int ry = -r.y; ry <= r.y; ++ry) {
          for (int rz = -r.z; rz <= r.z; ++rz) {
              POS_src_TYPE localPos = localMinPos + POS_src_INSTANCE(rx, ry, rz, 0);
              if( localPos.x >= 0 && localPos.y >= 0 && localPos.z >= 0) {
                const float value = (float) READ_IMAGE(src, sampler, localPos).x;
                if (value > localMin) {
                    localMin = value;
                    localMinPos = localPos;
                }
              }
          }
      }
  }

  IMAGE_dst_PIXEL_TYPE result = 0;
  if (r.z > 1) {  
    if (localMinPos.x == x && localMinPos.y == y && localMinPos.z == z) {
    result = 1;
    }
  }
  else if (r.y > 1) {  
    if (localMinPos.x == x && localMinPos.y == y) {
    result = 1;
    } 
  }
  else if (r.x > 1) {  
    if (localMinPos.x == x) {
    result = 1;
    }
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), result);
}
