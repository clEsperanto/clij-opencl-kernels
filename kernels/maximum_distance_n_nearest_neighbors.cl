const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void maximum_distance_n_nearest_neighbors(
    IMAGE_src_distance_matrix_TYPE  src_distance_matrix,
    IMAGE_dst_index_list_TYPE       dst_index_list,
    int                             nPoints
) 
{

  const int pointIndex = get_global_id(0);

  // so many point candidates are available:
  const int height = GET_IMAGE_HEIGHT(src_distance_matrix);

  float distances[1000];
  float indices[1000];

  int initialized_values = 0;

  // start at 1 to exclude background
  for (int y = 1; y < height; y++) {
    if (pointIndex != y) { // exclude distance to self
        float distance = READ_IMAGE(src_distance_matrix, sampler, POS_src_distance_matrix_INSTANCE(pointIndex, y, 0, 0)).x;

        if (initialized_values < nPoints) {
          initialized_values++;
          distances[initialized_values - 1] = distance;
          indices[initialized_values - 1] = y;
        }
        // sort by insert
        for (int i = initialized_values - 1; i >= 0; i--) {
            if (distance > distances[i]) {
                break;
            }
            if (distance < distances[i] && (i == 0 || distance >= distances[i - 1])) {
               for (int j = initialized_values - 1; j > i; j--) {
                    indices[j] = indices[j - 1];
                    distances[j] = distances[j - 1];
               }
               distances[i] = distance;
               indices[i] = y;
               break;
            }
        }
    }
  }

  float maximum = -1;

  for (int i = 0; i < initialized_values; i++) {
    if (distances[i] > maximum) {
      maximum = distances[i];
    }
  }

  WRITE_IMAGE(dst_index_list, POS_dst_index_list_INSTANCE(pointIndex, 0, 0, 0), CONVERT_dst_index_list_PIXEL_TYPE(maximum));
}