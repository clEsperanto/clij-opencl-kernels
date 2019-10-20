__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void binary_or_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_IN_3D  src2,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src1, sampler, pos).x);
  DTYPE_OUT value2 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src2, sampler, pos).x);
  if ( value1 > 0 || value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}

__kernel void binary_and_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_IN_3D  src2,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src1, sampler, pos).x);
  DTYPE_OUT value2 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src2, sampler, pos).x);
  if ( value1 > 0 && value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}

__kernel void binary_xor_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_IN_3D  src2,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src1, sampler, pos).x);
  DTYPE_OUT value2 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src2, sampler, pos).x);
  if ( (value1 > 0 && value2 == 0) || (value1 == 0 && value2 > 0)) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}

__kernel void binary_not_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = CONVERT_DTYPE_OUT(READ_IMAGE_3D(src1, sampler, pos).x);
  if ( value1 > 0) {
    value1 = 0;
  } else {
    value1 = 1;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}

__kernel void erode_box_neighborhood_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value != 0) {
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){x, y, z, 0})).x;
          if (value == 0) {
            break;
          }
        }
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

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void erode_box_neighborhood_slice_by_slice(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value != 0) {
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){x, y, z, 0})).x;
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

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void erode_diamond_neighborhood_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value != 0) {
    value = READ_IMAGE_3D(src, sampler, (pos + (int4){1, 0, 0, 0})).x;
    if (value != 0) {
      value = READ_IMAGE_3D(src, sampler, (pos + (int4){-1, 0, 0, 0})).x;
      if (value != 0) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 1, 0, 0})).x;
        if (value != 0) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, -1, 0, 0})).x;
          if (value != 0) {
            value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, 1, 0})).x;
            if (value != 0) {
              value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, -1, 0})).x;
            }
          }
        }
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void erode_diamond_neighborhood_slice_by_slice(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value != 0) {
    value = READ_IMAGE_3D(src, sampler, (pos + (int4){1, 0, 0, 0})).x;
    if (value != 0) {
      value = READ_IMAGE_3D(src, sampler, (pos + (int4){-1, 0, 0, 0})).x;
      if (value != 0) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 1, 0, 0})).x;
        if (value != 0) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, -1, 0, 0})).x;
        }
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void dilate_box_neighborhood_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value < 1) {
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){x, y, z, 0})).x;
          if (value != 0) {
            break;
          }
        }
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

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void dilate_box_neighborhood_slice_by_slice(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value == 0) {
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){x, y, z, 0})).x;
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

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void dilate_diamond_neighborhood_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value == 0) {

    value = READ_IMAGE_3D(src, sampler, (pos + (int4){1, 0, 0, 0})).x;
    if (value == 0) {
      value = READ_IMAGE_3D(src, sampler, (pos + (int4){-1, 0, 0, 0})).x;
      if (value == 0) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 1, 0, 0})).x;
        if (value == 0) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, -1, 0, 0})).x;
          if (value == 0) {
            value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, 1, 0})).x;
            if (value == 0) {
              value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, -1, 0})).x;
            }
          }
        }
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}

__kernel void dilate_diamond_neighborhood_slice_by_slice(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value == 0) {

    value = READ_IMAGE_3D(src, sampler, (pos + (int4){1, 0, 0, 0})).x;
    if (value == 0) {
      value = READ_IMAGE_3D(src, sampler, (pos + (int4){-1, 0, 0, 0})).x;
      if (value == 0) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 1, 0, 0})).x;
        if (value == 0) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, -1, 0, 0})).x;
          if (value == 0) {
            value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, 1, 0})).x;
            if (value == 0) {
              value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, -1, 0})).x;
            }
          }
        }
      }
    }
  }
  if (value != 0) {
    value = 1;
  }

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}
