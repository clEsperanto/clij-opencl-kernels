__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void convolve(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int kernelWidth  = GET_IMAGE_WIDTH(src1);
  const int kernelHeight = GET_IMAGE_HEIGHT(src1);
  const int kernelDepth  = GET_IMAGE_DEPTH(src1);

  const int ox = kernelWidth / 2;
  const int oy = kernelHeight / 2;
  const int oz = kernelDepth / 2;

  const POS_src0_TYPE pos_image  = POS_src0_INSTANCE( x,  y,  z, 0);
  const POS_src1_TYPE pos_kernel = POS_src1_INSTANCE(ox, oy, oz, 0);

  float sum = 0;
  for (int cz = -oz; cz <= oz; ++cz) {
    for (int cy = -oy; cy <= oy; ++cy) {
      for (int cx = -ox; cx <= ox; ++cx) {
        sum += (float) READ_IMAGE(src1, sampler, pos_kernel + POS_src1_INSTANCE(cx,cy,cz,0)).x 
             * (float) READ_IMAGE(src0, sampler, pos_image  + POS_src0_INSTANCE(cx,cy,cz,0)).x; 
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(sum));
}
