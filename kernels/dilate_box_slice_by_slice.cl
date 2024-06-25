__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void dilate_box_slice_by_slice(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int radius = 1;
  IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  if (value == 0) {
      for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
        value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0) + POS_src_INSTANCE(dx,dy,0,0)).x;
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

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
