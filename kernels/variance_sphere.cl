__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void variance_sphere(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       index0,
    const int       index1,
    const int       index2
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE coord = POS_src_INSTANCE(x,y,z,0);
  
  int4 radius = (int4){0,0,0,0};
  float4 squared = (float4){FLT_MIN, FLT_MIN, FLT_MIN, 0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = (index0-1)/2; squared.x = (float) (radius.x*radius.x);}
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = (index1-1)/2; squared.y = (float) (radius.y*radius.y);}
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = (index2-1)/2; squared.z = (float) (radius.z*radius.z);}

  int count = 0;
  float sum = 0;
  for (int dx = -radius.x; dx <= radius.x; dx++) {
    const float xSquared = dx * dx;
    for (int dy = -radius.y; dy <= radius.y; dy++) {
      const float ySquared = dy * dy;
      for (int dz = -radius.z; dz <= radius.z; dz++) {
        const float zSquared = dz * dz;
        if (xSquared / squared.x + ySquared / squared.y + zSquared / squared.z <= 1.0) {
          const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz,0);
          sum = sum + (float) READ_IMAGE(src, sampler, coord + pos).x;
          count++;
        }
      }
    }
  }
  const float mean_intensity = sum / count;
  sum = 0;
  count = 0;
  for (int dx = -radius.x; dx <= radius.x; ++dx) {
    const float xSquared = x * x;
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
      const float ySquared = y * y;
      for (int dz = -radius.z; dz <= radius.z; ++dz) {
        const float zSquared = z * z;
        if (xSquared / squared.x + ySquared / squared.y + zSquared / squared.z <= 1.0) {
          const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz,0);
          const float value = (float) READ_IMAGE(src, sampler, coord + pos).x;
          sum = sum + pow(value - mean_intensity, 2);
          count++;
        }
      }
    }
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(sum / (count)));
}