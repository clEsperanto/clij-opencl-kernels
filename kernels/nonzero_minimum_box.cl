__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void nonzero_minimum_box(
    IMAGE_src_TYPE   src,
    IMAGE_dst0_TYPE  dst0, 
    IMAGE_dst1_TYPE  dst1
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int4 r = (int4){0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { r.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { r.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { r.z = 1; }

  const POS_src_TYPE coord = POS_src_INSTANCE(x,y,z,0);
  IMAGE_src_PIXEL_TYPE foundMinimum = READ_IMAGE(src, sampler, coord).x;
  if (foundMinimum != 0) {
      IMAGE_src_PIXEL_TYPE originalValue = foundMinimum;
      for (int dx = -r.x; dx <= r.x; ++dx) {
        for (int dy = -r.y; dy <= r.y; ++dy) {
          for (int dz = -r.z; dz <= r.z; ++dz) {
            IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, coord + POS_src_INSTANCE(dx,dy,dz,0)).x;
            if ( value < foundMinimum && value > 0) {
              foundMinimum = value;
            }
          }
        }
      }
      
      if (foundMinimum != originalValue) {
        WRITE_IMAGE(dst0, POS_dst0_INSTANCE(0,0,0,0), 1);
      }
      WRITE_IMAGE(dst1, POS_dst1_INSTANCE(x,y,z,0), CONVERT_dst1_PIXEL_TYPE(foundMinimum));
  }
}
