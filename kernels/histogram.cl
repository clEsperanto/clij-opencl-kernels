// adapted code from https://github.com/bgaster/opencl-book-samples/blob/master/src/Chapter_14/histogram/histogram_image.cl

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void histogram(
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst,
    float minimum,
    float maximum,
    int step_size_x,
    int step_size_y,
    int step_size_z
)
{
    const int image_width  = GET_IMAGE_WIDTH(src);
    const int image_height = GET_IMAGE_HEIGHT(src);
    const int image_depth  = GET_IMAGE_DEPTH(src);
    
    const int y = get_global_id(0) * step_size_y;
    float range = maximum - minimum;

    uint tmp_histogram[NUMBER_OF_HISTOGRAM_BINS];
    for (int i = 0; i < NUMBER_OF_HISTOGRAM_BINS; ++i) {
        tmp_histogram[i] = 0;
    }

    for (int z = 0; z < GET_IMAGE_DEPTH(src); z += step_size_z) {
        for (int x = 0; x < GET_IMAGE_WIDTH(src); x += step_size_x) {
            float clr = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
            uint indx_x = convert_uint_sat( ( (clr - minimum) * (GET_IMAGE_WIDTH(dst) - 1) ) / range + 0.5);
            tmp_histogram[indx_x]++;
        }  
    }

    for (int idx = 0; idx < GET_IMAGE_WIDTH(dst); ++idx) {
        WRITE_IMAGE(dst, POS_dst_INSTANCE(idx,0,y,0), CONVERT_dst_PIXEL_TYPE(tmp_histogram[idx]));
    }
}
