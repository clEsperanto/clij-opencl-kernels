__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void euclidean_distance_from_label_centroid_map(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int w = GET_IMAGE_WIDTH(src0);
  const int h = GET_IMAGE_HEIGHT(src0);
  const int d = GET_IMAGE_DEPTH(src0);

  const int index = convert_int(READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x,y,z,0)).x);
  float distance = 0;
  if (index > 0) {
      float dx = 0, dy = 0, dz = 0;
      if (w > 1) {
        dx = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(index,0,0,0)).x;
      }
      if (h > 1) {
        dy = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(index,1,0,0)).x;
      }
      if (d > 1) {
        dz = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(index,2,0,0)).x;
      }
      const float distance_squared = (x-dx)*(x-dx) + (y-dy)*(y-dy) + (z-dz)*(z-dz) ;
      distance = sqrt(distance_squared);
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(distance));
}
