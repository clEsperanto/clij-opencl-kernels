__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void minimum_touching_neighbors (
    IMAGE_src_vector_TYPE  src_vector,
    IMAGE_src_matrix_TYPE  src_matrix,
    IMAGE_dst_TYPE         dst,
    int                    x_correction
) 
{
  const int label_id = get_global_id(0);
  const int label_count = get_global_size(0);

  bool initialized = false;
  float minimum = 0;

  int y = label_id;
  int x = 0;
  for (x = 0; x < label_id; x++) {
    float value = READ_IMAGE(src_matrix, sampler, POS_src_matrix_INSTANCE(x, y, 0, 0)).x;
    if (value > 0) {
      value = READ_IMAGE(src_vector, sampler, POS_src_vector_INSTANCE(x + x_correction, 0, 0, 0)).x;
      if (minimum > value || !initialized) {
        minimum = value;
        initialized = true;
      }
    }
  }

  x = label_id;
  for (y = label_id; y < label_count; y++) {
    float value = READ_IMAGE(src_matrix, sampler, POS_src_matrix_INSTANCE(x, y, 0, 0)).x;
    if (value > 0 || y == label_id) {
      value = READ_IMAGE(src_vector, sampler, POS_src_vector_INSTANCE(y + x_correction, 0, 0, 0)).x;
      if (minimum > value || !initialized) {
        minimum = value;
        initialized = true;
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(label_id, 0, 0, 0), CONVERT_dst_PIXEL_TYPE(minimum));
}

