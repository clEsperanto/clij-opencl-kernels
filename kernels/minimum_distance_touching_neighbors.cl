const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void minimum_distance_touching_neighbors (
    IMAGE_src_distance_matrix_TYPE  src_distance_matrix,
    IMAGE_src_touch_matrix_TYPE     src_touch_matrix,
    IMAGE_dst_index_list_TYPE       dst_index_list
) 
{
  const int label_id = get_global_id(0);
  const int label_count = get_global_size(0);

  int initialized = 0;
  float minimum = 0;

  int y = label_id;
  int x = 0;
  for (x = 1; x < label_id; x++) {
    POS_src_touch_matrix_TYPE pos = POS_src_touch_matrix_INSTANCE(x, y, 0, 0);
    float value = READ_IMAGE(src_touch_matrix, sampler, pos).x;
    if (value > 0) {
      value = READ_IMAGE(src_distance_matrix, sampler, pos).x;
      if (value < minimum || initialized == 0) {
        minimum = value;
        initialized = 1;
      }
    }
  }
  x = label_id;
  for (y = label_id + 1; y < label_count; y++) {
    POS_src_touch_matrix_TYPE pos = POS_src_touch_matrix_INSTANCE(x, y, 0, 0);
    float value = READ_IMAGE(src_touch_matrix, sampler, pos).x;
    if (value > 0) {
      value = READ_IMAGE(src_distance_matrix, sampler, pos).x;
      if (value < minimum || initialized == 0) {
        minimum = value;
        initialized = 1;
      }
    }
  }

  WRITE_IMAGE(dst_index_list, (POS_dst_index_list_INSTANCE(label_id, 0, 0, 0)), CONVERT_dst_index_list_PIXEL_TYPE(minimum));
}