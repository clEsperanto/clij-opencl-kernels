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

  const int ox = kernelWidth >> 1;  // Bit shift instead of division
  const int oy = kernelHeight >> 1;
  const int oz = kernelDepth >> 1;

  const POS_src0_TYPE coord_src0  = POS_src0_INSTANCE(x, y, z, 0);
  const POS_src1_TYPE coord_kernel = POS_src1_INSTANCE(ox, oy, oz, 0);

  float sum = 0.0f;
  
  // Pre-calculate loop bounds to avoid recalculation
  const int cz_start = -oz;
  const int cz_end = oz;
  const int cy_start = -oy;
  const int cy_end = oy;
  const int cx_start = -ox;
  const int cx_end = ox;
  
  for (int cz = cz_start; cz <= cz_end; ++cz) {
    const POS_src1_TYPE src1_z = POS_src1_INSTANCE(0, 0, cz, 0);
    const POS_src0_TYPE src0_z = POS_src0_INSTANCE(0, 0, cz, 0);
    
    for (int cy = cy_start; cy <= cy_end; ++cy) {
      const POS_src1_TYPE src1_yz = src1_z + POS_src1_INSTANCE(0, cy, 0, 0);
      const POS_src0_TYPE src0_yz = src0_z + POS_src0_INSTANCE(0, cy, 0, 0);
      
      for (int cx = cx_start; cx <= cx_end; ++cx) {
        // Accumulate position offsets to minimize arithmetic
        const POS_src1_TYPE pos_src1 = coord_kernel + src1_yz + POS_src1_INSTANCE(cx, 0, 0, 0);
        const POS_src0_TYPE pos_src0 = coord_src0 + src0_yz + POS_src0_INSTANCE(cx, 0, 0, 0);
        
        // Read once and multiply - fused multiply-add (FMA) opportunity
        const float src1_val = (float) READ_IMAGE(src1, sampler, pos_src1).x;
        const float src0_val = (float) READ_IMAGE(src0, sampler, pos_src0).x;
        sum = fma(src1_val, src0_val, sum);  // Use FMA for better performance
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(sum));
}
