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

  const int kernelWidth  = GET_IMAGE_WIDTH(src1)  > 1 ? GET_IMAGE_WIDTH(src1)  : 0;
  const int kernelHeight = GET_IMAGE_HEIGHT(src1) > 1 ? GET_IMAGE_HEIGHT(src1) : 0;
  const int kernelDepth  = GET_IMAGE_DEPTH(src1)  > 1 ? GET_IMAGE_DEPTH(src1)  : 0;

  const int4 c = (int4){kernelWidth / 2, kernelHeight / 2, kernelDepth / 2, 0};
  const POS_src0_TYPE pos_image  = POS_src0_INSTANCE(  x,  y,  z,0);
  const POS_src1_TYPE pos_kernel = POS_src1_INSTANCE(c.x,c.y,c.z,0);

  float sum = 0;
  for (int cz = -c.z; cz <= c.z; ++cz) {
    for (int cy = -c.y; cy <= c.y; ++cy) {
      for (int cx = -c.x; cx <= c.x; ++cx) {
        POS_src1_TYPE coord_kernel = pos_kernel + POS_src1_INSTANCE(cx,cy,cz,0);
        POS_src0_TYPE coord_image  = pos_image  + POS_src0_INSTANCE(cx,cy,cz,0);
        sum += (float) READ_IMAGE(src1, sampler, coord_kernel).x 
             * (float) READ_IMAGE(src0, sampler, coord_image ).x; 
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(sum));
}
