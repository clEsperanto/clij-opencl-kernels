
__kernel void max_project_3d_2d(
    DTYPE_IMAGE_OUT_2D dst_max,
    DTYPE_IMAGE_IN_3D src
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  DTYPE_IN max = 0;
  for(int z = 0; z < GET_IMAGE_IN_DEPTH(src); z++)
  {
    DTYPE_IN value = READ_IMAGE_3D(src,sampler,(int4)(x,y,z,0)).x;
    if (value > max || z == 0) {
      max = value;
    }
  }
  WRITE_IMAGE_2D(dst_max,(int2)(x,y), CONVERT_DTYPE_OUT(max));
}