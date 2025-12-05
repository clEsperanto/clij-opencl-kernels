__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gradient(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       axis // 0 -> x, 1 -> y, 2 -> z.
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Select the coordinate and its limit for the chosen axis.
    const int coord = (axis == 0) ? x : ((axis == 1) ? y : z);
    const int limit = (axis == 0) ? GET_IMAGE_WIDTH(src) : ((axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src));

    // Compute neighbor coordinates along the chosen axis with boundary clamping.
    const int forwardCoord = (coord < limit - 1) ? coord + 1 : coord;
    const int backwardCoord = (coord > 0) ? coord - 1 : coord;

    int fx = x, fy = y, fz = z;
    int bx = x, by = y, bz = z;

    if (axis == 0) {
        fx = forwardCoord; bx = backwardCoord;
    } else if (axis == 1) {
        fy = forwardCoord; by = backwardCoord;
    } else {
        fz = forwardCoord; bz = backwardCoord;
    }

    // Read neighbors.
    const float valueA = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(fx, fy, fz, 0)).x;
    const float valueB = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(bx, by, bz, 0)).x;

    const float norm = (coord == 0 || coord == limit - 1) ? 1.0f : 2.0f;
    const float gradientValue = (valueA - valueB) / norm;

    WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(gradientValue));
}
