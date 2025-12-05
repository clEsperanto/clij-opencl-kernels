//
//! ImageJ-specific kernel, DEPRECATED in clEsperanto
//

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void depth_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    IMAGE_lut_TYPE  lut,
    const float     scalar0,
    const float     scalar1
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
 
  float max = 0;
  float max_z = 0;
  for (float z = 0; z < GET_IMAGE_DEPTH(src); z += GET_IMAGE_DEPTH(src) / 255.0 ) {
    float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
    int condition = (value > max) | (z == 0);
    max = (condition * value) + ((1 - condition) * max);
    max_z = (condition * z) + ((1 - condition) * max_z);
  }

  float intensity = (max - scalar0) / (scalar1 - scalar0);
  float relative_z = max_z / (GET_IMAGE_DEPTH(src) - 1);

  intensity = clamp(intensity, 0.0f, 1.0f);
  relative_z = clamp(relative_z, 0.0f, 1.0f);

  const int index = (int) (255 * relative_z);
  const float r = (float) READ_IMAGE(lut, sampler, POS_lut_INSTANCE(index,0,0,0)).x * intensity;
  const float g = (float) READ_IMAGE(lut, sampler, POS_lut_INSTANCE(index,0,1,0)).x * intensity;
  const float b = (float) READ_IMAGE(lut, sampler, POS_lut_INSTANCE(index,0,2,0)).x * intensity;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,0,0), CONVERT_dst_PIXEL_TYPE(r));
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,1,0), CONVERT_dst_PIXEL_TYPE(g));
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,2,0), CONVERT_dst_PIXEL_TYPE(b));
}
