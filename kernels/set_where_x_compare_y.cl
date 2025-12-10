#ifndef COMPARISON_OP
    #error "COMPARISON_OP must be defined as a macro (e.g., #define COMPARISON_OP(x,y) (x > y))"
#endif

__kernel void set_where_x_compare_y(
    IMAGE_dst_TYPE  dst,
    const float     scalar
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    if (COMPARISON_OP(x, y)) {
        WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(scalar));
    }
}