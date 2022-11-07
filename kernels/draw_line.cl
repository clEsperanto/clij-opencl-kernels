
__kernel void draw_line(
    IMAGE_dst_TYPE  dst,
    const float     x1,
    const float     y1,
    const float     z1,
    const float     x2,
    const float     y2,
    const float     z2,
    const float     radius,
    const float     value
)
{
  const int dx = min(x1, x2) - radius + get_global_id(0);
  const int dy = min(y1, y2) - radius + get_global_id(1);
  const int dz = min(z1, z2) - radius + get_global_id(2);

  if (!((dx >= x1 - radius && dx <= x2 + radius) || (dx >= x2 - radius && dx <= x1 + radius))) {
    return;
  }
  if (!((dy >= y1 - radius && dy <= y2 + radius) || (dy >= y2 - radius && dy <= y1 + radius))) {
    return;
  }
  if (!((dz >= z1 - radius && dz <= z2 + radius) || (dz >= z2 - radius && dz <= z1 + radius))) {
    return;
  }

  // float4 r1 = (float4) {dx-x2, dy-y2, dz-z2, 0};
  // float4 r2 = (float4) {x1-x2, y1-y2, z1-z2, 0};
  const POS_dst_TYPE r1 = POS_dst_INSTANCE(dx-x2, dy-y2, dz-z2, 0);
  const POS_dst_TYPE r2 = POS_dst_INSTANCE(x1-x2, y1-y2, z1-z2, 0);
  const POS_dst_TYPE vector = cross(r2, r1);
  float distance = length(vector) / length(r2);

  if (distance < radius) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(dx, dy, dz, 0), CONVERT_dst_PIXEL_TYPE(value));
  }
}