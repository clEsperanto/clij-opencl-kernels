__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void local_cross_correlation(
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

  const int4 center = (int4){kernelWidth / 2, kernelHeight / 2, kernelDepth / 2, 0};
  const POS_src0_TYPE coord = POS_src0_INSTANCE(x, y, z, 0);

  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;

      for (int dz = -center.z; dz <= center.z; ++dz) {
    for (int dy = -center.y; dy <= center.y; ++dy) {
  for (int dx = -center.x; dx <= center.x; ++dx) {

        const POS_src1_TYPE coord_kernel = POS_src1_INSTANCE(dx + center.x, dy + center.y, dz + center.z, 0);
        const POS_src0_TYPE coord_image = coord + POS_src0_INSTANCE(dx, dy, dz, 0);

        const float Ia = (float) READ_IMAGE(src0, sampler, coord_image).x;
        const float Ib = (float) READ_IMAGE(src1, sampler, coord_kernel).x;

        // https://anomaly.io/understand-auto-cross-correlation-normalized-shift/index.html
        sum1 = sum1 + (Ia * Ib);
        sum2 = sum2 + (Ia * Ia);
        sum3 = sum3 + (Ib * Ib);
        }
    }
  }

  const float result = sum1 / sqrt((sum2 * sum3));
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(result));
}