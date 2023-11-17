__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void labelled_spots_to_point_list(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int index = ((int) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x) - 1;
  if (index >= 0) {
    if (GET_IMAGE_WIDTH(src) > 1) {  
      WRITE_IMAGE(dst, POS_dst_INSTANCE(index,0,0,0), CONVERT_dst_PIXEL_TYPE(x));
    }
    if (GET_IMAGE_HEIGHT(src) > 1) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(index,1,0,0), CONVERT_dst_PIXEL_TYPE(y));
    }
    if (GET_IMAGE_DEPTH(src) > 1) {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(index,2,0,0), CONVERT_dst_PIXEL_TYPE(z));
    }
  }
}



__kernel void labelled_spots_to_point_list (
    IMAGE_src_TYPE src,
    IMAGE_dst_point_list_TYPE dst_point_list
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int sx = get_global_id(0);
  const int sy = get_global_id(1);
  const int sz = get_global_id(2);

  const int index = ((int)READ_src_IMAGE(src, sampler, POS_src_INSTANCE(sx, sy, sz, 0)).x) - 1;
  if (index < 0) { // background pixel
    return;
  }

  int n_dimensions = GET_IMAGE_HEIGHT(dst_point_list);

  WRITE_dst_point_list_IMAGE(dst_point_list, POS_dst_point_list_INSTANCE(index, 0, 0, 0), CONVERT_dst_point_list_PIXEL_TYPE(sx));

  if (n_dimensions > 1) {
    WRITE_dst_point_list_IMAGE(dst_point_list, POS_dst_point_list_INSTANCE(index, 1, 0, 0), CONVERT_dst_point_list_PIXEL_TYPE(sy));
  }
  if (n_dimensions > 2) {
    WRITE_dst_point_list_IMAGE(dst_point_list, POS_dst_point_list_INSTANCE(index, 2, 0, 0), CONVERT_dst_point_list_PIXEL_TYPE(sz));
  }

}