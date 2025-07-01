__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gradient_y(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Cache image height to avoid repeated calls
    const int height = GET_IMAGE_HEIGHT(src);

    // Read current pixel value
    float centerValue = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;

    // Read neighboring pixel values with boundary checks
    float valueA = (y < height - 1) ? 
        (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y + 1, z, 0)).x : centerValue;
    float valueB = (y > 0) ? 
        (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y - 1, z, 0)).x : centerValue;

    // Compute gradient
    float norm = (y == 0 || y == height - 1) ? 1.0f : 2.0f;
    float gradient = (valueA - valueB) / norm;

    // Write result to output image
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(gradient));
}
