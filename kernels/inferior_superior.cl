
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define READ_3DIMAGE_ZERO_OUTSIDE(a,b,c) read_buffer3duc_zero_outside(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)
#define READ_2DIMAGE_ZERO_OUTSIDE(a,b,c) read_buffer2duc_zero_outside(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)

inline uchar2 read_buffer2duc_zero_outside(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global uchar * buffer_var, sampler_t sampler, int2 position )
{
    int2 pos = (int2){position.x, position.y};
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (uchar2){0, 0};
    }
    return (uchar2){buffer_var[pos_in_buffer],0};
}

inline uchar2 read_buffer3duc_zero_outside(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global uchar * buffer_var, sampler_t sampler, int4 position )
{
    int4 pos = (int4){position.x, position.y, position.z, 0};
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (uchar2){0, 0};
    }
    return (uchar2){buffer_var[pos_in_buffer],0};
}


inline void inferior_superior_3d (
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x, y, z, 0};

  // if value is already 0, erode will return 0
  float value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, pos).x;
  if (value != 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P0
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, j, 0, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P1
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, 0, j, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P2
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){0, i, j, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P3
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, j, j, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P4
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){j, i, -i, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P5
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, j, i, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P6
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, j, -i, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P7
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, i, j, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P8
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_3DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int4){i, -i, j, 0})).x;
        if (value != 0) {
          break;
        }
      }
      if (value != 0) {
        break;
      }
    }
  if (value == 0) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // If all erodes are 0 then return 0
  WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
}

inline void inferior_superior_2d (
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  // if value is already 1, dilate will return 1
  float value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, pos).x;
  if (value == 1) {
    WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  /* Dilate with kernel [[1, 0, 0], 
                         [0, 1, 0], 
                         [0, 0, 1]] */
  value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){1, 1})).x;
  if (value == 0) {
    value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){-1, -1})).x;
    if (value == 0) {
      WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
      return;
    }
  }

  /* Dilate with kernel [[0, 1, 0], 
                         [0, 1, 0], 
                         [0, 1, 0]] */
  value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){0, 1})).x;
    if (value == 0) {
      value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){0, -1})).x;
      if (value == 0) {
        WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
        return;
      }
    }

  /* Dilate with kernel [[0, 0, 1], 
                         [0, 1, 0], 
                         [1, 0, 0]] */
  value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){-1, 1})).x;
    if (value == 0) {
      value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){1, -1})).x;
      if (value == 0) {
        WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
        return;
      }
    }

  /* Dilate with kernel [[0, 0, 0], 
                         [1, 1, 1], 
                         [0, 0, 0]] */
  value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){1, 0})).x;
    if (value == 0) {
      value = READ_2DIMAGE_ZERO_OUTSIDE(src, sampler, (pos + (int2){-1, 0})).x;
      if (value == 0) {
        WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
        return;
      }
    }

  // If all dilates are 1 then return 1
  WRITE_dst_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
}

__kernel void inferior_superior(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  if (GET_IMAGE_DEPTH(src) > 1) {
    inferior_superior_3d(src, dst);
  } else {
    inferior_superior_2d(src, dst);
  }
}