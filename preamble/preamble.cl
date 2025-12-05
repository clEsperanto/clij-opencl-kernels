#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

// #pragma OPENCL EXTENSION cl_amd_printf : enable

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifndef M_PI
    #define   M_PI 3.14159265358979323846f /* pi */
#endif

#ifndef M_LOG2E
    #define   M_LOG2E   1.4426950408889634074f /* log_2 e */
#endif
 
#ifndef M_LOG10E
    #define   M_LOG10E   0.43429448190325182765f /* log_10 e */
#endif
 
#ifndef M_LN2
    #define   M_LN2   0.69314718055994530942f  /* log_e 2 */
#endif

#ifndef M_LN10
    #define   M_LN10   2.30258509299404568402f /* log_e 10 */
#endif

#ifndef BUFFER_READ_WRITE
    #define BUFFER_READ_WRITE 1

#define DEFINE_READ_BUFFER3D(SUFFIX, TYPE) \
    inline TYPE##2 read_buffer3d ## SUFFIX(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global TYPE * buffer_var, sampler_t sampler, int4 position) { \
        int4 pos = (int4){position.x, position.y, position.z, 0}; \
        pos.x = clamp(pos.x, 0, read_buffer_width - 1); \
        pos.y = clamp(pos.y, 0, read_buffer_height - 1); \
        pos.z = clamp(pos.z, 0, read_buffer_depth - 1); \
        int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height; \
        if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) return (TYPE##2){0, 0}; \
        return (TYPE##2){buffer_var[pos_in_buffer], 0}; \
    }

#define DEFINE_WRITE_BUFFER3D(SUFFIX, TYPE) \
    inline void write_buffer3d ## SUFFIX(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global TYPE * buffer_var, int4 pos, TYPE value) { \
        int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height; \
        if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) return; \
        buffer_var[pos_in_buffer] = value; \
    }

#if defined(USE_3D) && defined(USE_CHAR)
DEFINE_READ_BUFFER3D(c, char)
DEFINE_WRITE_BUFFER3D(c, char)
#endif

#if defined(USE_3D) && defined(USE_UCHAR)
DEFINE_READ_BUFFER3D(uc, uchar)
DEFINE_WRITE_BUFFER3D(uc, uchar)
#endif

#if defined(USE_3D) && defined(USE_SHORT)
DEFINE_READ_BUFFER3D(s, short)
DEFINE_WRITE_BUFFER3D(s, short)
#endif

#if defined(USE_3D) && defined(USE_USHORT)
DEFINE_READ_BUFFER3D(us, ushort)
DEFINE_WRITE_BUFFER3D(us, ushort)
#endif

#if defined(USE_3D) && defined(USE_INT)
DEFINE_READ_BUFFER3D(i, int)
DEFINE_WRITE_BUFFER3D(i, int)
#endif

#if defined(USE_3D) && defined(USE_UINT)
DEFINE_READ_BUFFER3D(ui, uint)
DEFINE_WRITE_BUFFER3D(ui, uint)
#endif

// #if defined(USE_3D) && defined(USE_LONG)
// DEFINE_READ_BUFFER3D(l, long)
// DEFINE_WRITE_BUFFER3D(l, long)
// #endif

// #if defined(USE_3D) && defined(USE_ULONG)
// DEFINE_READ_BUFFER3D(ul, ulong)
// DEFINE_WRITE_BUFFER3D(ul, ulong)
// #endif

#if defined(USE_3D) && defined(USE_FLOAT)
DEFINE_READ_BUFFER3D(f, float)
DEFINE_WRITE_BUFFER3D(f, float)
#endif

#define DEFINE_READ_BUFFER2D(SUFFIX, TYPE) \
    inline TYPE##2 read_buffer2d ## SUFFIX(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global TYPE * buffer_var, sampler_t sampler, int2 position) { \
        int2 pos = (int2){position.x, position.y}; \
        pos.x = clamp(pos.x, 0, read_buffer_width - 1); \
        pos.y = clamp(pos.y, 0, read_buffer_height - 1); \
        int pos_in_buffer = pos.x + pos.y * read_buffer_width; \
        if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) return (TYPE##2){0, 0}; \
        return (TYPE##2){buffer_var[pos_in_buffer], 0}; \
    }

#define DEFINE_WRITE_BUFFER2D(SUFFIX, TYPE) \
    inline void write_buffer2d ## SUFFIX(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global TYPE * buffer_var, int2 pos, TYPE value) { \
        int pos_in_buffer = pos.x + pos.y * write_buffer_width; \
        if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) return; \
        buffer_var[pos_in_buffer] = value; \
    }

#if defined(USE_2D) && defined(USE_CHAR)
DEFINE_READ_BUFFER2D(c, char)
DEFINE_WRITE_BUFFER2D(c, char)
#endif

#if defined(USE_2D) && defined(USE_UCHAR)
DEFINE_READ_BUFFER2D(uc, uchar)
DEFINE_WRITE_BUFFER2D(uc, uchar)
#endif

#if defined(USE_2D) && defined(USE_SHORT)
DEFINE_READ_BUFFER2D(s, short)
DEFINE_WRITE_BUFFER2D(s, short)
#endif

#if defined(USE_2D) && defined(USE_USHORT)
DEFINE_READ_BUFFER2D(us, ushort)
DEFINE_WRITE_BUFFER2D(us, ushort)
#endif

#if defined(USE_2D) && defined(USE_INT)
DEFINE_READ_BUFFER2D(i, int)
DEFINE_WRITE_BUFFER2D(i, int)
#endif

#if defined(USE_2D) && defined(USE_UINT)
DEFINE_READ_BUFFER2D(ui, uint)
DEFINE_WRITE_BUFFER2D(ui, uint)
#endif

// #if defined(USE_2D) && defined(USE_LONG)
// DEFINE_READ_BUFFER2D(l, long)
// DEFINE_WRITE_BUFFER2D(l, long)
// #endif

// #if defined(USE_2D) && defined(USE_ULONG)
// DEFINE_READ_BUFFER2D(ul, ulong)
// DEFINE_WRITE_BUFFER2D(ul, ulong)
// #endif

#if defined(USE_2D) && defined(USE_FLOAT)
DEFINE_READ_BUFFER2D(f, float)
DEFINE_WRITE_BUFFER2D(f, float)
#endif

#define DEFINE_READ_BUFFER1D(SUFFIX, TYPE) \
    inline TYPE##2 read_buffer1d ## SUFFIX(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global TYPE * buffer_var, sampler_t sampler, int position) { \
        int pos = clamp(position, 0, read_buffer_width - 1); \
        if (pos < 0 || pos >= read_buffer_width) return (TYPE##2){0, 0}; \
        return (TYPE##2){buffer_var[pos], 0}; \
    }

#define DEFINE_WRITE_BUFFER1D(SUFFIX, TYPE) \
    inline void write_buffer1d ## SUFFIX(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global TYPE * buffer_var, int pos, TYPE value) { \
        if (pos < 0 || pos >= write_buffer_width) return; \
        buffer_var[pos] = value; \
    }

#if defined(USE_1D) && defined(USE_CHAR)
DEFINE_READ_BUFFER1D(c, char)
DEFINE_WRITE_BUFFER1D(c, char)
#endif

#if defined(USE_1D) && defined(USE_UCHAR)
DEFINE_READ_BUFFER1D(uc, uchar)
DEFINE_WRITE_BUFFER1D(uc, uchar)
#endif

#if defined(USE_1D) && defined(USE_SHORT)
DEFINE_READ_BUFFER1D(s, short)
DEFINE_WRITE_BUFFER1D(s, short)
#endif

#if defined(USE_1D) && defined(USE_USHORT)
DEFINE_READ_BUFFER1D(us, ushort)
DEFINE_WRITE_BUFFER1D(us, ushort)
#endif

#if defined(USE_1D) && defined(USE_INT)
DEFINE_READ_BUFFER1D(i, int)
DEFINE_WRITE_BUFFER1D(i, int)
#endif

#if defined(USE_1D) && defined(USE_UINT)
DEFINE_READ_BUFFER1D(ui, uint)
DEFINE_WRITE_BUFFER1D(ui, uint)
#endif

// #if defined(USE_1D) && defined(USE_LONG)
// DEFINE_READ_BUFFER1D(l, long)
// DEFINE_WRITE_BUFFER1D(l, long)
// #endif

// #if defined(USE_1D) && defined(USE_ULONG)
// DEFINE_READ_BUFFER1D(ul, ulong)
// DEFINE_WRITE_BUFFER1D(ul, ulong)
// #endif

#if defined(USE_1D) && defined(USE_FLOAT)
DEFINE_READ_BUFFER1D(f, float)
DEFINE_WRITE_BUFFER1D(f, float)
#endif

inline uchar clij_convert_uchar_sat(float value) {
    return (uchar)clamp(value, 0.0f, 255.0f);
}

inline char clij_convert_char_sat(float value) {
    return (char)clamp(value, -128.0f, 127.0f);
}

inline ushort clij_convert_ushort_sat(float value) {
    return (ushort)clamp(value, 0.0f, 65535.0f);
}

inline short clij_convert_short_sat(float value) {
    return (short)clamp(value, -32768.0f, 32767.0f);
}

inline uint clij_convert_uint_sat(float value) {
    return (uint)clamp(value, 0.0f, 4294967295.0f);
}

inline int clij_convert_int_sat(float value) {
    return (int)clamp(value, -2147483648.0f, 2147483647.0f);
}

// inline ulong clij_convert_ulong_sat(float value) {
//     return (ulong)clamp(value, 0.0f, 18446744073709551615.0f);
// }

// inline long clij_convert_long_sat(float value) {
//     return (long)clamp(value, -9223372036854775808.0f, 9223372036854775807.0f);
// }

inline float clij_convert_float_sat(float value) {
    return value;
}

#define READ_IMAGE(a,b,c) READ_ ## a ## _IMAGE(a,b,c)
#define WRITE_IMAGE(a,b,c) WRITE_ ## a ## _IMAGE(a,b,c)

#endif
