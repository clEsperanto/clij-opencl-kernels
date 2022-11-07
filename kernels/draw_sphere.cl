
__kernel void draw_sphere(
    IMAGE_dst_TYPE  dst,
    const float     cx,
    const float     cy,
    const float     cz,
    const float     rx,
    const float     ry,
    const float     rz,
    const float     rxsq,
    const float     rysq,
    const float     rzsq,
    const float     value
)
{
  const float x = get_global_id(0);
  const float y = get_global_id(1);
  const float z = get_global_id(2);

  if ((x < cx - rx) || (x > cx + rx)) {
    return;
  }
  if ((y < cy - ry) || (y > cy + ry)) {
    return;
  }
  if ((z < cz - rz) || (z > cz + rz)) {
    return;
  }

  float xSquared = pow(x - cx, 2);
  float ySquared = pow(y - cy, 2);
  float zSquared = pow(z - cz, 2);

  if ((xSquared / rxsq + ySquared / rysq + zSquared / rzsq) <= 1.0) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(value));
  }
}