__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void kuwahara(
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst,
    const int radius,
    const float sigma
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    const int width  = GET_IMAGE_WIDTH(src);
    const int height = GET_IMAGE_HEIGHT(src);
    const int depth  = GET_IMAGE_DEPTH(src);

    if (x >= width || y >= height || z >= depth) {
        return;
    }

    // If the image has depth > 1, process it as a 3D stack (8 octants),
    // otherwise fall back to the 2D behaviour (4 quadrants).
    const bool is_3d = depth > 1;
    const int num_regions = is_3d ? 8 : 4;

    const int radius_z = is_3d ? radius : 0;

    const float two_sigma_sq = 2.0f * sigma * sigma;

    // Regions are enumerated with a 3-bit index:
    // bit 0 -> x side (0: dx <= 0, 1: dx >= 0)
    // bit 1 -> y side (0: dy <= 0, 1: dy >= 0)
    // bit 2 -> z side (0: dz <= 0, 1: dz >= 0), unused in 2D
    // Sign lookup tables replace recomputing (q & bit) ? 1 : -1 on every voxel.
    const int SIGN_X[8] = {-1,  1, -1,  1, -1,  1, -1,  1};
    const int SIGN_Y[8] = {-1, -1,  1,  1, -1, -1,  1,  1};
    const int SIGN_Z[8] = {-1, -1, -1, -1,  1,  1,  1,  1};

    float mean[8]       = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float variance[8]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float weight_sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq[8]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Single pass: accumulate weighted sum and sum of squares per region.
    for (int dz = -radius_z; dz <= radius_z; dz++) {
        int pz = clamp(z + dz, 0, depth - 1);
        for (int dy = -radius; dy <= radius; dy++) {
            int py = clamp(y + dy, 0, height - 1);
            for (int dx = -radius; dx <= radius; dx++) {
                int px = clamp(x + dx, 0, width - 1);

                float dist_sq = (float)(dx * dx + dy * dy + dz * dz);
                float w = exp(-dist_sq / two_sigma_sq);
                float val = READ_IMAGE(src, sampler, POS_src_INSTANCE(px, py, pz, 0)).x;
                float wv = w * val;

                for (int q = 0; q < num_regions; q++) {
                    if (dx * SIGN_X[q] >= 0 && dy * SIGN_Y[q] >= 0 && dz * SIGN_Z[q] >= 0) {
                        mean[q] += wv;
                        sum_sq[q] += wv * val;
                        weight_sum[q] += w;
                    }
                }
            }
        }
    }

    for (int q = 0; q < num_regions; q++) {
        if (weight_sum[q] > 0.0f) {
            mean[q] /= weight_sum[q];
            variance[q] = fmax(0.0f, sum_sq[q] / weight_sum[q] - mean[q] * mean[q]);
        }
    }

    // Blend region means, weighting by inverse variance
    // (small epsilon avoids division by zero)
    const float epsilon = 1e-6f;
    float total_inv_var = 0.0f;
    float inv_var[8];
    for (int q = 0; q < num_regions; q++) {
        inv_var[q] = 1.0f / (variance[q] + epsilon);
        total_inv_var += inv_var[q];
    }

    float result = 0.0f;
    for (int q = 0; q < num_regions; q++) {
        result += mean[q] * (inv_var[q] / total_inv_var);
    }

    WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(result));
}