__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void addWeightedPixelwise_3d(DTYPE_IMAGE_IN_3D  src,
                                 DTYPE_IMAGE_IN_3D  src1,
                                 float factor,
                                 float factor1,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x * factor + READ_IMAGE_3D(src1, sampler, pos).x * factor1);

  WRITE_IMAGE_3D (dst, pos, value);
}