__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void gradientX_2d
(
  DTYPE_IMAGE_OUT_2D dst, DTYPE_IMAGE_IN_2D src
)
{
  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i,j};
  const int2 coordA = (int2){i-1,j};
  const int2 coordB = (int2){i+1,j};

  DTYPE_IN valueA = (DTYPE_OUT)READ_IMAGE_2D(src, sampler, coordA).x;
  DTYPE_IN valueB = (DTYPE_OUT)READ_IMAGE_2D(src, sampler, coordB).x;
  DTYPE_OUT res = CONVERT_DTYPE_OUT(valueB - valueA);

  WRITE_IMAGE_2D(dst, coord, res);
}


__kernel void gradientY_2d
(
  DTYPE_IMAGE_OUT_2D dst, DTYPE_IMAGE_IN_2D src
)
{
  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i,j};
  const int2 coordA = (int2){i,j-1};
  const int2 coordB = (int2){i,j+1};

  DTYPE_IN valueA = (DTYPE_OUT)READ_IMAGE_2D(src, sampler, coordA).x;
  DTYPE_IN valueB = (DTYPE_OUT)READ_IMAGE_2D(src, sampler, coordB).x;
  DTYPE_OUT res = CONVERT_DTYPE_OUT(valueB - valueA);

  WRITE_IMAGE_2D(dst, coord, res);
}

__kernel void gradientX_3d
(
  DTYPE_IMAGE_OUT_3D dst, DTYPE_IMAGE_IN_3D src
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  const int4 coord  = (int4){i, j, k, 0};
  const int4 coordA = (int4){i-1, j, k, 0};
  const int4 coordB = (int4){i+1, j, k, 0};

  DTYPE_IN valueA = (DTYPE_OUT)READ_IMAGE_3D(src, sampler, coordA).x;
  DTYPE_IN valueB = (DTYPE_OUT)READ_IMAGE_3D(src, sampler, coordB).x;
  DTYPE_OUT res = CONVERT_DTYPE_OUT(valueB - valueA);

  WRITE_IMAGE_3D(dst, coord, res);
}

__kernel void gradientY_3d
(
  DTYPE_IMAGE_OUT_3D dst, DTYPE_IMAGE_IN_3D src
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  const int4 coord  = (int4){i, j, k, 0};
  const int4 coordA = (int4){i, j-1, k, 0};
  const int4 coordB = (int4){i, j+1, k, 0};

  DTYPE_IN valueA = (DTYPE_OUT)READ_IMAGE_3D(src, sampler, coordA).x;
  DTYPE_IN valueB = (DTYPE_OUT)READ_IMAGE_3D(src, sampler, coordB).x;
  DTYPE_OUT res = CONVERT_DTYPE_OUT(valueB - valueA);

  WRITE_IMAGE_3D(dst, coord, res);
}

__kernel void gradientZ_3d
(
  DTYPE_IMAGE_OUT_3D dst, DTYPE_IMAGE_IN_3D src
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  const int4 coord  = (int4){i, j, k, 0};
  const int4 coordA = (int4){i, j, k-1, 0};
  const int4 coordB = (int4){i, j, k+1, 0};

  DTYPE_IN valueA = (DTYPE_OUT)READ_IMAGE_3D(src, sampler, coordA).x;
  DTYPE_IN valueB = (DTYPE_OUT)READ_IMAGE_3D(src, sampler, coordB).x;
  DTYPE_OUT res = CONVERT_DTYPE_OUT(valueB - valueA);

  WRITE_IMAGE_3D(dst, coord, res);
}

