
__kernel void draw_box(
    IMAGE_dst_TYPE  dst,
    const float     x1,
    const float     y1,
    const float     z1,
    const float     x2,
    const float     y2,
    const float     z2,
    const float     value
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  if (!((x >= x1 && x <= x2) || (x >= x2 && x <= x1))) {
    return;
  }
  if (!((y >= y1 && y <= y2) || (y >= y2 && y <= y1))) {
    return;
  }
  if (!((z >= z1 && z <= z2) || (z >= z2 && z <= z1))) {
    return;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(value));
}