__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void laplace_diamond(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  float result = 0;
  float weight = 0;
  if(GET_IMAGE_WIDTH(src) > 1) {
    result += (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE( 1,0,0,0)).x * -1.0 
    result += (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(-1,0,0,0)).x * -1.0 ;
    weight += 2;
  }
  if(GET_IMAGE_HEIGHT(src) > 1) {
    result += (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0, 1,0,0)).x * -1.0 
    result += (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,-1,0,0)).x * -1.0 ;
    weight += 2;
  }
  if(GET_IMAGE_DEPTH(src) > 1) {
    result += (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,0, 1,0)).x * -1.0 
    result += (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,0,-1,0)).x * -1.0 ;
    weight += 2;
  }
  result += (float) READ_IMAGE(src, sampler, pos).x * weight;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(result));
}