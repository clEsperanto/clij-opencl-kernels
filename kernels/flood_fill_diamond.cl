__constant sampler_t sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void flood_fill_diamond(
    IMAGE_src_TYPE   src,
    IMAGE_dst0_TYPE  dst0,
    IMAGE_dst1_TYPE  dst1,
    const float      scalar0,
    const float      scalar1
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
  if (value != scalar0) { 
    WRITE_IMAGE(dst1, pos, CONVERT_dst_PIXEL_TYPE(value));
    return; 
  }

  bool replace = false;
  if (GET_IMAGE_WIDTH(src) > 1)
  {
    if (READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(-1,0,0,0)).x == scalar1 ) {
      replace = true;
    }
    if (!replace && READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(1,0,0,0)).x == scalar1 ) {
      replace = true;
    }
  }
  if (GET_IMAGE_HEIGHT(src) > 1)
  {
    if (!replace && READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,-1,0,0)).x == scalar1 ) {
      replace = true;
    }
    if (!replace && READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,1,0,0)).x == scalar1 ) {
      replace = true;
    }
  }
  if (GET_IMAGE_DEPTH(src) > 1) {
      if (!replace && READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,0,-1,0)).x == scalar1 ) {
        replace = true;
      }
      if (!replace && READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(0,0,1,0)).x == scalar1 ) {
        replace = true;
      }
  }

  if (replace) {
    WRITE_IMAGE(dst0, POS_dst0_INSTANCE(0,0,0,0), 1);
    WRITE_IMAGE(dst1, POS_dst1_INSTANCE(x,y,z,0), CONVERT_dst1_PIXEL_TYPE(scalar1));
  } else {
    WRITE_IMAGE(dst1, POS_dst1_INSTANCE(x,y,z,0), CONVERT_dst1_PIXEL_TYPE(value));
  }
}
