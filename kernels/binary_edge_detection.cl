__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void binary_edge_detection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  IMAGE_src_PIXEL_TYPE valueToWrite = READ_IMAGE(src, sampler, pos).x;
  if (valueToWrite != 0) {
    valueToWrite = 0;
    IMAGE_src_PIXEL_TYPE value = 0;
    if (GET_IMAGE_WIDTH(src) > 1 && valueToWrite == 0 ){  
      value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(1,0,0,0))).x;
      if ( value == 0) {
          valueToWrite = 1;
      } 
      else {
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(-1,0,0,0))).x;
        if ( value == 0) {
            valueToWrite = 1;
        } 
      }
    }
    if (GET_IMAGE_HEIGHT(src) > 1 && valueToWrite == 0 ){  
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,1,0,0))).x;
        if ( value == 0) {
          valueToWrite = 1;
        } 
        else {
          value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,-1,0,0))).x;
          if ( value == 0) {
            valueToWrite = 1;
          } 
        }
    }
    if (GET_IMAGE_DEPTH(src) > 1 && valueToWrite == 0 ){  
      value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,0,1,0))).x;
      if ( value == 0) {
        valueToWrite = 1;
      } 
      else {
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,0,-1,0))).x;
        if ( value == 0) {
          valueToWrite = 1;
        }
      }
    }
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(valueToWrite));
}
