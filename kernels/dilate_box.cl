__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void dilate_box(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  int4 r = (int4){0,0,0,0};
  if (GET_IMAGE_DEPTH(src)  > 1) { r.z = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { r.y = 1; }
  if (GET_IMAGE_WIDTH(src)  > 1) { r.x = 1; }

  float value = READ_IMAGE(src, sampler, pos).x;
  if (value < 1) {
        for (int dz = -r.z; dz <= r.z; ++dz) {
      for (int dy = -r.y; dy <= r.y; ++dy) {
    for (int dx = -r.x; dx <= r.x; ++dx) {
          value = READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(dx,dy,dz,0)).x;
          if (value != 0) {
            break;
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
  }

  if (value != 0) {
    value = 1;
  }
  WRITE_IMAGE (dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
