__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void dilate_box(
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

  const int4 r = (int4){ (GET_IMAGE_WIDTH(src)  > 1) * ((scalar0 - 1) / 2), 
                         (GET_IMAGE_HEIGHT(src) > 1) * ((scalar1 - 1) / 2), 
                         (GET_IMAGE_DEPTH(src)  > 1) * ((scalar2 - 1) / 2), 0 };

  IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
  if (value != 0)
  {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

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

  WRITE_IMAGE (dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value != 0));
}
