__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#define XY 0
#define XZ 1
#define YZ 2

#if TRANSPOSE_MODE == XY // X <-> Y
  #define SRC_X y
  #define SRC_Y x
  #define SRC_Z z
#elif TRANSPOSE_MODE == XZ // X <-> Z
  #define SRC_X z
  #define SRC_Y y
  #define SRC_Z x
#elif TRANSPOSE_MODE == YZ // Y <-> Z
  #define SRC_X x
  #define SRC_Y z
  #define SRC_Z y
#else
  #error "TRANSPOSE_MODE must be XY, XZ, or YZ"
#endif

__kernel void transpose(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE spos = POS_src_INSTANCE(SRC_X, SRC_Y, SRC_Z, 0);
  const POS_dst_TYPE dpos = POS_dst_INSTANCE(x, y, z, 0);

  const float value = READ_IMAGE(src, sampler, spos).x;
  WRITE_IMAGE(dst, dpos, CONVERT_dst_PIXEL_TYPE(value));
}
