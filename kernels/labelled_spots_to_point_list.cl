__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void labelled_spots_to_point_list(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int w = GET_IMAGE_WIDTH(src);
  const int h = GET_IMAGE_HEIGHT(src);
  const int z = GET_IMAGE_DEPT(src);

  if( x > w || y > h || z > d) { return; } // cuda wrong block/grid get outside of coord

  const int index = ((int) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x) - 1;
  if (index >= 0) {
    if (w > 1) {  
      WRITE_IMAGE(dst, POS_dst_INSTANCE(index,0,0,0), CONVERT_dst_PIXEL_TYPE(x));
    }
    if (h > 1) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(index,1,0,0), CONVERT_dst_PIXEL_TYPE(y));
    }
    if (z > 1) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(index,2,0,0), CONVERT_dst_PIXEL_TYPE(z));
    }
  }
}
