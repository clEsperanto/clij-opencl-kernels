__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void minimum_sphere(
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

  int4 radius = (int4){0,0,0,0};
  float4 squared = (float4){FLT_MIN,FLT_MIN,FLT_MIN,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = (scalar0-1)/2; squared.x = convert_float(radius.x*radius.x);}
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = (scalar1-1)/2; squared.y = convert_float(radius.y*radius.y);}
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = (scalar2-1)/2; squared.z = convert_float(radius.z*radius.z);}

  IMAGE_src_PIXEL_TYPE minimumValue = READ_IMAGE(src, sampler, coord + POS_src_INSTANCE(x,y,z,0)).x;
  for (int dx = -radius.x; dx <= radius.x; ++dx) {
    const float xSquared = dx * dx;
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
      const float ySquared = dy * dy;
      for (int dz = -radius.z; dz <= radius.z; ++dz) {
        const float zSquared = dz * dz;
        if (xSquared / squared.x + ySquared / squared.y + zSquared / squared.z <= 1.0) {
          const IMAGE_src_PIXEL_TYPE value_res = READ_IMAGE(src, sampler, coord + POS_src_INSTANCE(dx,dy,dz,0)).x;
          if (value_res < minimumValue) {
            minimumValue = value_res;
          }
        }
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(minimumValue));
}
