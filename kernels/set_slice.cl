__kernel void set_slice(
    IMAGE_dst_TYPE  dst,
    const int       dimension, // 0: row, 1: column, 2: plane
    const int       index,
    const float     scalar
)
{
    const int3 gid = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    const int3 coords = (dimension == 0) ? (int3)(gid.x, index, gid.z) :
                        (dimension == 1) ? (int3)(index, gid.y, gid.z) :
                                           (int3)(gid.x, gid.y, index);
    WRITE_IMAGE(dst, POS_dst_INSTANCE(coords.x, coords.y, coords.z, 0), CONVERT_dst_PIXEL_TYPE(scalar));
}