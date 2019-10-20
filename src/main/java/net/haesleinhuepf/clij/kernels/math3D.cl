__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void multiplyPixelwise_3d(DTYPE_IMAGE_IN_3D  src,
                                   DTYPE_IMAGE_IN_3D src1,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x * READ_IMAGE_3D(src1, sampler, pos).x);

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void dividePixelwise_3d(DTYPE_IMAGE_IN_3D  src,
                                 DTYPE_IMAGE_IN_3D  src1,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x / READ_IMAGE_3D(src1, sampler, pos).x);

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void multiplySliceBySliceWithScalars(DTYPE_IMAGE_IN_3D  src,
                                 __constant    float*  scalars,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos3d = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos3d).x * scalars[z]);

  WRITE_IMAGE_3D (dst, pos3d, value);
}


__kernel void addPixelwise_3d(DTYPE_IMAGE_IN_3D  src,
                                 DTYPE_IMAGE_IN_3D  src1,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x + READ_IMAGE_3D(src1, sampler, pos).x);

  WRITE_IMAGE_3D (dst, pos, value);
}


__kernel void addScalar_3d(DTYPE_IMAGE_IN_3D  src,
                                 float scalar,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x + scalar);

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void multiplyScalar_3d(DTYPE_IMAGE_IN_3D  src,
                                 float scalar,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x * scalar);

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void maxPixelwise_3d(DTYPE_IMAGE_IN_3D src,
                              DTYPE_IMAGE_IN_3D src1,
                              DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_IN input = READ_IMAGE_3D(src, sampler, pos).x;
  const DTYPE_IN input1 = READ_IMAGE_3D(src1, sampler, pos).x;

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(max(input, input1));

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void minPixelwise_3d(DTYPE_IMAGE_IN_3D src,
                              DTYPE_IMAGE_IN_3D src1,
                              DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_IN input = READ_IMAGE_3D(src, sampler, pos).x;
  const DTYPE_IN input1 = READ_IMAGE_3D(src1, sampler, pos).x;

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(min(input, input1));

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void maxPixelwiseScalar_3d(DTYPE_IMAGE_IN_3D src,
                              float valueB,
                              DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_IN input = READ_IMAGE_3D(src, sampler, pos).x;
  const DTYPE_IN input1 = valueB;

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(max(input, input1));

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void minPixelwiseScalar_3d(DTYPE_IMAGE_IN_3D src,
                              float valueB,
                              DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_IN input = READ_IMAGE_3D(src, sampler, pos).x;
  const DTYPE_IN input1 = valueB;

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(min(input, input1));

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void power_3d(DTYPE_IMAGE_IN_3D src,
                              DTYPE_IMAGE_OUT_3D dst,
                              float exponent
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_IN input = READ_IMAGE_3D(src, sampler, pos).x;

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(pow(input, exponent));

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void multiply_pixelwise_with_coordinate_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst,
                          int dimension
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  const DTYPE_OUT value = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src, sampler, pos).x * get_global_id(dimension));

  WRITE_IMAGE_3D (dst, pos, value);
}




