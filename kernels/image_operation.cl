__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#ifndef APPLY_OP
  #error "APPLY_OP must be defined as a macro (e.g., #define APPLY_OP(x,y) (x+y))"
#endif

__kernel void image_operation(
    IMAGE_src0_TYPE src0,
    IMAGE_src1_TYPE src1,
    IMAGE_dst_TYPE  dst
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    const float value0 = (float) READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x,y,z,0)).x;
    const float value1 = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x,y,z,0)).x;
    const float res = APPLY_OP(value0, value1);

    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(res));
}