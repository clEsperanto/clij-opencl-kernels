__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| SAMPLER_ADDRESS | SAMPLER_FILTER;

__kernel void generate_touch_matrix(
    IMAGE_src_TYPE src
    IMAGE_dst_TYPE dst,
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int w = GET_IMAGE_WIDTH(src);
  const int h = GET_IMAGE_HEIGHT(src);
  const int d = GET_IMAGE_DEPTH(src);

  float label = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  if (x <= w - 1) {
    const float labelx = READ_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y, z, 0)).x;
    if (label != labelx) {
      WRITE_IMAGE(dst, (POS_dst_INSTANCE(label, labelx, 0, 0)), CONVERT_dst_PIXEL_TYPE(1));
      WRITE_IMAGE(dst, (POS_dst_INSTANCE(labelx, label, 0, 0)), CONVERT_dst_PIXEL_TYPE(1));
    }
  }
  if (y <= h - 1) {
    const float labely = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y + 1, z, 0)).x;
    if (label != labely) {
      WRITE_IMAGE(dst, (POS_dst_INSTANCE(label, labely, 0, 0)), CONVERT_dst_PIXEL_TYPE(1));
      WRITE_IMAGE(dst, (POS_dst_INSTANCE(labely, label, 0, 0)), CONVERT_dst_PIXEL_TYPE(1));
    }
  }
  if (z <= d - 1) {
    const float labelz = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z + 1, 0)).x;
    if (label != labelz) {
      WRITE_IMAGE(dst, (POS_dst_INSTANCE(label, labelz, 0, 0)), CONVERT_dst_PIXEL_TYPE(1));
      WRITE_IMAGE(dst, (POS_dst_INSTANCE(labelz, label, 0, 0)), CONVERT_dst_PIXEL_TYPE(1));
    }
  }
}