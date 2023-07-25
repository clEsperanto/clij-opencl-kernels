__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void maximum_octagon_box(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  int4 radius = (int4){0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = 1; }

  IMAGE_src_PIXEL_TYPE maximum = READ_IMAGE(src, sampler, pos).x;
  for (int dx = -radius.x; dx <= radius.x; ++dx) {
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
      for (int dz = -radius.z; dz <= radius.z; ++dz) {
        IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(dx,dy,dz,0))).x;
        if (maximum < value) {
          maximum = value;
        }
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(maximum));
}
