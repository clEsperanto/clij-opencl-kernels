__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void multiplyStackWithPlanePixelwise(DTYPE_IMAGE_IN_3D  src,
                                 DTYPE_IMAGE_IN_2D  src1,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos3d = (int4){x,y,z,0};
  const int2 pos2d = (int2){x,y};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos3d).x * READ_IMAGE_2D(src1, sampler, pos2d).x);

  WRITE_IMAGE_3D (dst, pos3d, value);
}


