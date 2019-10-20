__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void absolute_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  float value = READ_IMAGE_2D(src, sampler, pos).x;
  if ( value < 0 ) {
    value = -1 * value;
  }

  WRITE_IMAGE_2D (dst, pos, CONVERT_DTYPE_OUT(value));
}