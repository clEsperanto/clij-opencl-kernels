__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void count_touching_neighbors(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{
  const int label_id = get_global_id(0);
  const int label_count = GET_IMAGE_WIDTH(src);
  IMAGE_dst_PIXEL_TYPE count = 0;

  int y = label_id;
  int x = 0;
  for (x = 0; x < label_id; ++x) {
    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,0,0)).x;
    if (value > 0) {
      count++;
    }
  }
  x = label_id;
  for (y = label_id + 1; y < label_count; ++y) {
    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,0,0)).x;
    if (value > 0) {
      count++;
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(label_id,0,0,0), count);
}
