  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void label_spots_in_x(
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst,
    IMAGE_countX_TYPE countX,
    IMAGE_countXY_TYPE countXY
) 
{
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  if (y >= GET_IMAGE_HEIGHT(dst)) return;
  if (z >= GET_IMAGE_DEPTH(dst)) return;

  int startingIndex = 0;
  for (int iz = 0; iz < z; iz++) {
    startingIndex = startingIndex + READ_IMAGE(countXY, sampler, POS_countXY_INSTANCE(iz, 0, 0, 0)).x;
  }
  for (int iy = 0; iy < y; iy++) {
    startingIndex = startingIndex + READ_IMAGE(countX, sampler, POS_countX_INSTANCE(z, iy, 0, 0)).x;
  }
  for(int x = 0; x < GET_IMAGE_WIDTH(src); x++)
  {
    float value = READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
    if (value != 0) {
      startingIndex++;
      WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(startingIndex));
    }
  }
}
