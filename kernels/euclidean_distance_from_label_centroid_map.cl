__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void euclidean_distance_from_label_centroid_map(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const int w = GET_IMAGE_WIDTH(src0);
  const int h = GET_IMAGE_HEIGHT(src0);
  const int d = GET_IMAGE_DEPTH(src0);

  const int index = convert_int(READ_IMAGE(src0, sampler, POS_src0_INSTANCE(i,j,k,0)).x);
  float distance = 0;
  if (index > 0) {
      const float dx = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(index,0,0,0)).x;
      const float dy = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(index,1,0,0)).x;

      float dz = 0;
      if (d > 1) {
        dz = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(index,2,0,0)).x;
      }
      const float distance_squared = (i-dx)*(i-dx) + (j-dy)*(j-dy) + (k-dz)*(k-dz) ;
      distance = sqrt(distance_squared);
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(i,j,k,0), CONVERT_dst_PIXEL_TYPE(distance));
}
