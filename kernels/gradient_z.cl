__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gradient_z(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Cache image depth to avoid repeated calls
    const int depth = GET_IMAGE_DEPTH(src);

    // Read current pixel value
    float centerValue = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;

    // Read neighboring pixel values with boundary checks
    float valueA = (z < depth - 1) ? 
        (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z + 1, 0)).x : centerValue;
    float valueB = (z > 0) ? 
        (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z - 1, 0)).x : centerValue;

    // Compute gradient
    float norm = (z > 0 && z < depth - 1) ? 2.0f : 1.0f;
    float gradient = (valueA - valueB) / norm;

    // Write result to output image
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(gradient));
}
