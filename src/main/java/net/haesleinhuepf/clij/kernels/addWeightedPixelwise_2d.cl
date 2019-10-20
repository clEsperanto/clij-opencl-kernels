__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void addWeightedPixelwise_2d(DTYPE_IMAGE_IN_2D  src,
                                 DTYPE_IMAGE_IN_2D  src1,
                                 float factor,
                                 float factor1,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src, sampler, pos).x * factor + READ_IMAGE_2D(src1, sampler, pos).x * factor1);

  WRITE_IMAGE_2D (dst, pos, value);
}