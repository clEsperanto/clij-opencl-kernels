__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void read_intensities_from_positions(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
    const int i = get_global_id(0);

    const int x = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x, 0, 0, 0)).x;
    const int y = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x, 1, 0, 0)).x;
    const int z = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x, 2, 0, 0)).x;
    const float intensity = (float) READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x, y, z, 0)).x;
    WRITE_IMAGE(dst, POS_dst_INSTANCE(i, 0, 0, 0), CONVERT_dst_PIXEL_TYPE(intensity));
}