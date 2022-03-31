__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void onlyzero_overwrite_maximum_box(
  IMAGE_src_TYPE   src,
  IMAGE_dst0_TYPE  dst0,
  IMAGE_dst1_TYPE  dst1,
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  int4 radius = (int4) {0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = 1; }

  const IMAGE_src_PIXEL_TYPE originalValue = READ_IMAGE(src, sampler, pos).x;
  IMAGE_src_PIXEL_TYPE foundMaximum = originalValue;
  if (foundMaximum == 0) {
    for (int dx = -radius.x; dx <= radius.x; ++dx) {
      for (int dy = -radius.y; dy <= radius.y; ++dy) {
        for (int dz = -radius.z; dz <= radius.z; ++dz) {
          IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(dx,dy,dz,0))).x;
          if (value > foundMaximum) {
            foundMaximum = value;
          }
        }
      }
    }
  }
  if (foundMaximum != originalValue) {
      WRITE_IMAGE(dst0, POS_dst0_INSTANCE(0,0,0,0), 1);
  }
  else {
      WRITE_IMAGE(dst0, POS_dst0_INSTANCE(0,0,0,0), 0);
  }
  WRITE_IMAGE(dst1, POS_dst1_INSTANCE(x,y,z,0), CONVERT_dst1_PIXEL_TYPE(foundMaximum));
}
