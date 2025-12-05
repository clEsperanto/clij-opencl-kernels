__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void dilate_sphere(
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
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  const int4 radius = (int4){(GET_IMAGE_WIDTH(src) > 1 && scalar0 > 1) * ((scalar0-1)/2), 
                             (GET_IMAGE_HEIGHT(src) > 1 && scalar1 > 1) * ((scalar1-1)/2),
                             (GET_IMAGE_DEPTH(src) > 1 && scalar2 > 1) * ((scalar2-1)/2), 
                             0};
  const float4 squared = (float4){(radius.x > 0) ? (float)(radius.x*radius.x) : FLT_MIN,
                                  (radius.y > 0) ? (float)(radius.y*radius.y) : FLT_MIN,
                                  (radius.z > 0) ? (float)(radius.z*radius.z) : FLT_MIN,
                                  0};

  IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
  if (value != 0)
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  for (int dz = -radius.z; dz <= radius.z; dz++) {
    const float zSquared = dz * dz;
    for (int dy = -radius.y; dy <= radius.y; dy++) {
      const float ySquared = dy * dy;
      for (int dx = -radius.x; dx <= radius.x; dx++) {
        const float xSquared = dx * dx;
        if (xSquared / squared.x + ySquared / squared.y + zSquared / squared.z <= 1.0) {

          value = READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(dx,dy,dz,0)).x;
          if (value != 0) {
            break;
          }

        }
      }
      if (value != 0) {
        break;
      }
    }
    if (value != 0) {
      break;
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value != 0));
}