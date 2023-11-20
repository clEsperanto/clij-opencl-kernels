__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void std_z_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int depth = GET_IMAGE_DEPTH(src);

  float sum = 0;
  int count = 0;
  for(int z = 0; z < depth; z++)
  {
    sum = sum + (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
    count++;
  }
  float mean = (sum / count);

  sum = 0;
  for(int z = 0; z < depth; z++)
  {
    float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x - mean;
    sum = sum + (value * value);
  }

  const float std_value = sqrt((float) sum / (count - 1));
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,0,0), CONVERT_dst_PIXEL_TYPE(std_value));
}