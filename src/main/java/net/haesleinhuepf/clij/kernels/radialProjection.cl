
__kernel void radialProjection3d(
    DTYPE_IMAGE_OUT_3D dst,
    DTYPE_IMAGE_IN_3D src,
    float deltaAngle
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const float imageHalfWidth = GET_IMAGE_IN_WIDTH(src) / 2;
  const float imageHalfHeight = GET_IMAGE_IN_HEIGHT(src) / 2;

  float angleInRad = ((float)z) * deltaAngle / 180.0 * M_PI;
  //float maxRadius = sqrt(pow(imageHalfWidth, 2.0f) + pow(imageHalfHeight, 2.0f));
  float radius = x;

  const int sx = (int)(imageHalfWidth + sin(angleInRad) * radius);
  const int sy = (int)(imageHalfHeight + cos(angleInRad) * radius);
  const int sz = y;

  DTYPE_IN value = READ_IMAGE_3D(src,sampler,(int4)(sx,sy,sz,0)).x;
  WRITE_IMAGE_3D(dst,(int4)(x,y,z,0), CONVERT_DTYPE_OUT(value));
}