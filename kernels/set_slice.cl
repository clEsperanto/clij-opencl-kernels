__kernel void set_slice(
    IMAGE_dst_TYPE  dst,
    const int       dimension, // 0: row, 1: column, 2: plane
    const int       index,
    const float     scalar
)
{
    const int3 gid = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    const int3 coords = (int3)(
        gid.x * (dimension != 0) + index * (dimension == 0),
        gid.y * (dimension != 1) + index * (dimension == 1),
        gid.z * (dimension != 2) + index * (dimension == 2)
    );
    WRITE_IMAGE(dst, POS_dst_INSTANCE(coords.x, coords.y, coords.z, 0), CONVERT_dst_PIXEL_TYPE(scalar));
}