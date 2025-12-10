__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void mode_sphere
(
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

  int4 coord = (int4){x,y,z,0};
  const int4 radius = (int4){(GET_IMAGE_WIDTH(src) > 1 && scalar0 > 1) * ((scalar0-1)/2), 
                             (GET_IMAGE_HEIGHT(src) > 1 && scalar1 > 1) * ((scalar1-1)/2),
                             (GET_IMAGE_DEPTH(src) > 1 && scalar2 > 1) * ((scalar2-1)/2), 
                             0};
  const float4 squared = (float4){(radius.x > 0) ? (float)(radius.x*radius.x) : FLT_MIN,
                                  (radius.y > 0) ? (float)(radius.y*radius.y) : FLT_MIN,
                                  (radius.z > 0) ? (float)(radius.z*radius.z) : FLT_MIN,
                                  0};

  long histogram[256];
  for (int h = 0; h < 256; h++){
    histogram[h]=0;
  }

  for (int dz = -radius.z; dz <= radius.z; ++dz) {
    const float zSquared = dz * dz;
    const int x3 = coord.z + dz;
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
      const float ySquared = dy * dy;
      const int x2 = coord.y + dy;
      for (int dx = -radius.x; dx <= radius.x; ++dx) {
        const float xSquared = dx * dx;
        const int x1 = coord.x + dx;
        if (xSquared / squared.x + ySquared / squared.y + zSquared / squared.z <= 1.0) {
          
          if (x1 < 0 || x2 < 0 || x3 < 0 || x1 >= GET_IMAGE_WIDTH(src) || x2 >= GET_IMAGE_HEIGHT(src) || x3 >= GET_IMAGE_DEPTH(src)) {
            continue;
          }
          
          const POS_src_TYPE pos = POS_src_INSTANCE(x1,x2,x3,0);
          const int value_res = (int) READ_IMAGE(src, sampler, pos).x;
          histogram[value_res]++;
        }
      }
    }
  }

  long max_value = 0;
  int max_pos = 0;
  for (int h = 0; h < 256; h++){
    if (max_value < histogram[h]){
      max_value = histogram[h];
      max_pos = h;
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(max_pos));
}