__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_minima(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{ 
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 radius = (int4){GET_IMAGE_WIDTH(src) > 1, GET_IMAGE_HEIGHT(src) > 1, GET_IMAGE_DEPTH(src) > 1, 0};

  int isMin = 1;
  float localMin = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  const int4 pos = (int4){x,y,z,0};

  for (int dz = -radius.z; dz <= radius.z; ++dz) {
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
      for (int dx = -radius.x; dx <= radius.x; ++dx) {
        int4 localPos = pos + (int4){dx,dy,dz,0};
        if( localPos.x == pos.x && localPos.y == pos.y && localPos.z == pos.z) {
          continue;
        }
        if( localPos.x < 0 || localPos.y < 0 || localPos.z < 0 || localPos.x >= GET_IMAGE_WIDTH(src) || localPos.y >= GET_IMAGE_HEIGHT(src) || localPos.z >= GET_IMAGE_DEPTH(src) ) {
          continue;
        }
        const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(localPos.x,localPos.y,localPos.z,0)).x;
        if (value <= localMin) {
            isMin = 0;
            break;
        }
      }
      if (!isMin) {
        break;
      }
    }
    if (!isMin) {
      break;
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), isMin);
}
