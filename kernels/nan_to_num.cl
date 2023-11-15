  __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void nan_to_num(
    IMAGE_dst_TYPE  dst,
    IMAGE_src_TYPE  src,
    float           nan,
    float           pinf,
    float           ninf
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  if (isnan(nan))  { nan = 0; }
  if (isinf(pinf)) { pinf = FLT_MAX; }
  if (isinf(ninf)) { ninf = -FLT_MAX; }

  float value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  if (isnan(value)) { value = nan; }
  if (isinf(value) && value > 0) { value = pinf; }
  if (isinf(value) && value < 0) { value = ninf; }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z,0), CONVERT_dst_PIXEL_TYPE(value));
}