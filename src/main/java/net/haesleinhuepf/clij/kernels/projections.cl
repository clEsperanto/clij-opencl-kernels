__kernel void sum_project_3d_2d(
    DTYPE_IMAGE_OUT_2D dst,
    DTYPE_IMAGE_IN_3D src
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  DTYPE_IN sum = 0;
  for(int z = 0; z < GET_IMAGE_IN_DEPTH(src); z++)
  {
    sum = sum + READ_IMAGE_3D(src,sampler,(int4)(x,y,z,0)).x;
  }
  WRITE_IMAGE_2D(dst,(int2)(x,y), CONVERT_DTYPE_OUT(sum));
}


__kernel void mean_project_3d_2d(
    DTYPE_IMAGE_OUT_2D dst,
    DTYPE_IMAGE_IN_3D src
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  DTYPE_IN sum = 0;
  int count = 0;
  for(int z = 0; z < GET_IMAGE_IN_DEPTH(src); z++)
  {
    sum = sum + READ_IMAGE_3D(src,sampler,(int4)(x,y,z,0)).x;
    count++;
  }
  WRITE_IMAGE_2D(dst,(int2)(x,y), CONVERT_DTYPE_OUT(sum / count));
}


__kernel void min_project_3d_2d(
    DTYPE_IMAGE_OUT_2D dst_min,
    DTYPE_IMAGE_IN_3D src
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  DTYPE_IN min = 0;
  for(int z = 0; z < GET_IMAGE_IN_DEPTH(src); z++)
  {
    DTYPE_IN value = READ_IMAGE_3D(src,sampler,(int4)(x,y,z,0)).x;
    if (value < min || z == 0) {
      min = value;
    }
  }
  WRITE_IMAGE_2D(dst_min,(int2)(x,y), CONVERT_DTYPE_OUT(min));
}


__kernel void max_project_dim_select_3d_2d(
    DTYPE_IMAGE_OUT_2D dst_max,
    DTYPE_IMAGE_IN_3D src,
    int projection_x,
    int projection_y,
    int projection_dim
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  int4 sourcePos = (int4)(0,0,0,0);
  int2 targetPos = (int2)(get_global_id(0), get_global_id(1));

  if (projection_x == 0) {
    sourcePos.x = get_global_id(0);
  } else if (projection_x == 1) {
    sourcePos.y = get_global_id(0);
  } else {
    sourcePos.z = get_global_id(0);
  }
  if (projection_y == 0) {
    sourcePos.x = get_global_id(1);
  } else if (projection_y == 1) {
    sourcePos.y = get_global_id(1);
  } else {
    sourcePos.z = get_global_id(1);
  }


  int max_d = 0;
  if (projection_dim == 0) {
    max_d = GET_IMAGE_IN_WIDTH(src);
  } else if (projection_dim == 1) {
    max_d = GET_IMAGE_IN_HEIGHT(src);
  } else {
    max_d = GET_IMAGE_IN_DEPTH(src);
  }

  DTYPE_IN max = 0;
  for(int d = 0; d < max_d; d++)
  {
    if (projection_dim == 0) {
      sourcePos.x = d;
    } else if (projection_dim == 0) {
      sourcePos.y = d;
    } else {
      sourcePos.z = d;
    }
    DTYPE_IN value = READ_IMAGE_3D(src,sampler, sourcePos).x;
    if (value > max || d == 0) {
      max = value;
    }
  }
  WRITE_IMAGE_2D(dst_max,targetPos, CONVERT_DTYPE_OUT(max));
}



