__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void nonzero_minimum_diamond(
    IMAGE_src_TYPE   src,
    IMAGE_dst0_TYPE  dst0,
    IMAGE_dst1_TYPE  dst1 
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  IMAGE_src_PIXEL_TYPE foundMinimum = READ_IMAGE(src, sampler, pos).x;
  if (foundMinimum != 0) {
    IMAGE_src_PIXEL_TYPE originalValue = foundMinimum;
    IMAGE_src_PIXEL_TYPE value = 0;
    
    if(GET_IMAGE_WIDTH(src) > 1) {
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(1,0,0,0))).x;
        if ( value < foundMinimum && value > 0) {
            foundMinimum = value;
        }
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(-1,0,0,0))).x;
        if ( value < foundMinimum && value > 0) {
            foundMinimum = value;
        }
    }
    if(GET_IMAGE_HEIGHT(src) > 1) {
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,1,0,0))).x;
        if ( value < foundMinimum && value > 0) {
            foundMinimum = value;
        }
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,-1,0,0))).x;
        if ( value < foundMinimum && value > 0) {
            foundMinimum = value;
        }
    }
    if(GET_IMAGE_DEPTH(src) > 1) {
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,0,1,0))).x;
        if ( value < foundMinimum && value > 0) {
            foundMinimum = value;
        }
        value = READ_IMAGE(src, sampler, (pos + POS_src_INSTANCE(0,0,-1,0))).x;
        if ( value < foundMinimum && value > 0) {
            foundMinimum = value;
        }
    }
    
    if (foundMinimum != originalValue) {
        WRITE_IMAGE(dst0, POS_dst0_INSTANCE(0,0,0,0), 1);
    }
    else {
        WRITE_IMAGE(dst0, POS_dst0_INSTANCE(0,0,0,0), 0);
    }
    WRITE_IMAGE(dst1, POS_dst1_INSTANCE(x,y,z,0), CONVERT_dst1_PIXEL_TYPE(foundMinimum));
  }
}
