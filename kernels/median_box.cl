__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline void sort(IMAGE_dst_PIXEL_TYPE * array, const int array_size)
{
    IMAGE_dst_PIXEL_TYPE temp;
    for(int i = 0; i < array_size; i++) {
        int j;
        temp = array[i];
        for(j = i - 1; j >= 0 && temp < array[j]; j--) {
            array[j+1] = array[j];
        }
        array[j+1] = temp;
    }
}

inline IMAGE_dst_PIXEL_TYPE median(IMAGE_dst_PIXEL_TYPE * array, const int array_size)
{
    sort(array, array_size);
    return array[array_size / 2];
}


__kernel void median_box(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       scalar0,
    const int       scalar1,
    const int       scalar2
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int4 radius = (int4){0,0,0,0};
  float4 squared = (float4){FLT_MIN, FLT_MIN, FLT_MIN, 0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = (scalar0-1)/2; squared.x = (float) (radius.x*radius.x);}
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = (scalar1-1)/2; squared.y = (float) (radius.y*radius.y);}
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = (scalar2-1)/2; squared.z = (float) (radius.z*radius.z);}
  const POS_src_TYPE coord = POS_src_INSTANCE(x,y,z,0);

  // int array_size = scalar0 * scalar1 * scalar2;
  IMAGE_dst_PIXEL_TYPE array[MAX_ARRAY_SIZE];

  int count = 0;
  for (int dx = -radius.x; dx <= radius.x; dx++) {
    for (int dy = -radius.y; dy <= radius.y; dy++) {
      for (int dz = -radius.z; dz <= radius.z; dz++) {
        const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz, 0);
        IMAGE_src_PIXEL_TYPE value_res = READ_IMAGE(src, sampler, coord + pos).x;
        array[count] = CONVERT_dst_PIXEL_TYPE(value_res);
        count++;
      }
    }
  }
  // array_size = count;
  // = copyBoxVolumeNeighborhoodToArray(src, array, coord, Nx, Ny, Nz);

  IMAGE_dst_PIXEL_TYPE res = median(array, count);
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), res);
}

