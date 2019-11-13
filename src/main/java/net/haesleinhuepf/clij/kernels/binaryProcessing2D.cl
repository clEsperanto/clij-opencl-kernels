__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void binary_or_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_IN_2D  src2,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src1, sampler, pos).x);
  DTYPE_OUT value2 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src2, sampler, pos).x);
  if ( value1 > 0 || value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}

__kernel void binary_and_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_IN_2D  src2,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src1, sampler, pos).x);
  DTYPE_OUT value2 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src2, sampler, pos).x);
  if ( value1 > 0 && value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}

__kernel void binary_xor_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_IN_2D  src2,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src1, sampler, pos).x);
  DTYPE_OUT value2 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src2, sampler, pos).x);
  if ( (value1 > 0 && value2 == 0) || (value1 == 0 && value2 > 0)) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}

__kernel void binary_not_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_2D(src1, sampler, pos).x);
  if ( value1 > 0) {
    value1 = 0;
  } else {
    value1 = 1;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}


__kernel void erode_box_neighborhood_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  float value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value != 0) {
    for (int ax = -1; ax <= 1; ax++) {
      for (int ay = -1; ay <= 1; ay++) {
        value = READ_IMAGE_2D(src, sampler, (pos + (int2){ax, ay})).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_2D (dst, pos, CONVERT_DTYPE_OUT(value));
}


__kernel void erode_diamond_neighborhood_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  float value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value != 0) {
    value = READ_IMAGE_2D(src, sampler, (pos + (int2){1, 0})).x;
    if (value != 0) {
      value = READ_IMAGE_2D(src, sampler, (pos + (int2){-1, 0})).x;
      if (value != 0) {
        value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, 1})).x;
        if (value != 0) {
          value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, -1})).x;
        }
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_2D (dst, pos, CONVERT_DTYPE_OUT(value));
}


__kernel void dilate_box_neighborhood_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  float value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value == 0) {
    for (int ax = -1; ax <= 1; ax++) {
      for (int ay = -1; ay <= 1; ay++) {
        value = READ_IMAGE_2D(src, sampler, (pos + (int2){ax, ay})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_2D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void dilate_diamond_neighborhood_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  float value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value == 0) {
    value = READ_IMAGE_2D(src, sampler, (pos + (int2){1, 0})).x;
    if (value == 0) {
      value = READ_IMAGE_2D(src, sampler, (pos + (int2){-1, 0})).x;
      if (value == 0) {
        value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, 1})).x;
        if (value == 0) {
          value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, -1})).x;
        }
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_2D (dst, pos, CONVERT_DTYPE_OUT(value));
}
