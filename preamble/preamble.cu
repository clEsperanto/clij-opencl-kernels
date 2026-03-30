
#define MINMAX_TYPE int
#define sampler_t int

#define FLT_MIN 1.19209e-07
#define FLT_MAX 1e+37

#define MAX_ARRAY_SIZE 1000

#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int
#define ulong unsigned long


__device__ inline float saturate(float x, float minval, float maxval) {
    return fminf(fmaxf(x, minval), maxval);
}

__device__ inline int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

__device__ inline int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}

__device__ inline int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__device__ inline float pow(float x, int y) {
    return pow(float(x), float(y));
}

__device__ inline float2 sqrt(float2 a) {
    return make_float2(sqrt(a.x), sqrt(a.y));
}

__device__ inline float4 cross(float4 a, float4 b)
{ 
    return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0); 
}

__device__ inline float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ inline float length(float4 v)
{
    return sqrtf(dot(v, v));
}

__device__ inline unsigned int atomic_add(unsigned int* address, unsigned int value) {
    return atomicAdd(address, value);
}

__device__ inline uchar clij_convert_uchar_sat(float value) {
    return (uchar) saturate(value, 0.0f, 255.0f);
}

__device__ inline char clij_convert_char_sat(float value) {
    return (char) saturate(value, -128.0f, 127.0f);
}

__device__ inline ushort clij_convert_ushort_sat(float value) {    
    return (ushort) saturate(value, 0.0f, 65535.0f);
}

__device__ inline short clij_convert_short_sat(float value) {
    return (short) saturate(value, -32768.0f, 32767.0f);
}

__device__ inline uint clij_convert_uint_sat(float value) {
    return (uint) saturate(value, 0.0f, 4294967295.0f);
}

__device__ inline int clij_convert_int_sat(float value) {
    return (int) saturate(value, -2147483648.0f, 2147483647.0f);
}

__device__ inline uint clij_convert_ulong_sat(float value) {
    return (ulong) saturate(value, 0.0f, 18446744073709551615.0f);
}

__device__ inline int clij_convert_long_sat(float value) {
    return (long) saturate(value, -9223372036854775808.0f, 9223372036854775807.0f);
}

__device__ inline float clij_convert_float_sat(float value) {
    return value;
}

#define READ_IMAGE(a,b,c) READ_ ## a ## _IMAGE(a,b,c)
#define WRITE_IMAGE(a,b,c) WRITE_ ## a ## _IMAGE(a,b,c)

#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH
#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT
#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH

#define CLK_NORMALIZED_COORDS_FALSE 1
#define CLK_ADDRESS_CLAMP_TO_EDGE 2
#define CLK_FILTER_NEAREST 4
#define CLK_NORMALIZED_COORDS_TRUE 8
#define CLK_ADDRESS_CLAMP 16
#define CLK_FILTER_LINEAR 32
#define CLK_ADDRESS_NONE 64

#ifndef BUFFER_READ_WRITE
    #define BUFFER_READ_WRITE 1

#define DEFINE_READ_BUFFER3D(SUFFIX, TYPE, MAKE_FUNC) \
    __device__ inline TYPE##2 read_buffer3d ## SUFFIX(int read_buffer_width, int read_buffer_height, int read_buffer_depth, TYPE * buffer_var, int sampler, int4 position) { \
        int4 pos = make_int4(position.x, position.y, position.z, 0); \
        pos.x = max((MINMAX_TYPE)pos.x, (MINMAX_TYPE)0); \
        pos.y = max((MINMAX_TYPE)pos.y, (MINMAX_TYPE)0); \
        pos.z = max((MINMAX_TYPE)pos.z, (MINMAX_TYPE)0); \
        pos.x = min((MINMAX_TYPE)pos.x, (MINMAX_TYPE)read_buffer_width - 1); \
        pos.y = min((MINMAX_TYPE)pos.y, (MINMAX_TYPE)read_buffer_height - 1); \
        pos.z = min((MINMAX_TYPE)pos.z, (MINMAX_TYPE)read_buffer_depth - 1); \
        int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height; \
        if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) return MAKE_FUNC(0, 0); \
        return MAKE_FUNC(buffer_var[pos_in_buffer], 0); \
    }

#define DEFINE_WRITE_BUFFER3D(SUFFIX, TYPE) \
    __device__ inline void write_buffer3d ## SUFFIX(int write_buffer_width, int write_buffer_height, int write_buffer_depth, TYPE * buffer_var, int4 pos, TYPE value) { \
        int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height; \
        if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) return; \
        buffer_var[pos_in_buffer] = value; \
    }

#if defined(USE_3D) && defined(USE_CHAR)
DEFINE_READ_BUFFER3D(c, char, make_char2)
DEFINE_WRITE_BUFFER3D(c, char)
#endif

#if defined(USE_3D) && defined(USE_UCHAR)
DEFINE_READ_BUFFER3D(uc, uchar, make_uchar2)
DEFINE_WRITE_BUFFER3D(uc, uchar)
#endif

#if defined(USE_3D) && defined(USE_SHORT)
DEFINE_READ_BUFFER3D(s, short, make_short2)
DEFINE_WRITE_BUFFER3D(s, short)
#endif

#if defined(USE_3D) && defined(USE_USHORT)
DEFINE_READ_BUFFER3D(us, ushort, make_ushort2)
DEFINE_WRITE_BUFFER3D(us, ushort)
#endif

#if defined(USE_3D) && defined(USE_INT)
DEFINE_READ_BUFFER3D(i, int, make_int2)
DEFINE_WRITE_BUFFER3D(i, int)
#endif

#if defined(USE_3D) && defined(USE_UINT)
DEFINE_READ_BUFFER3D(ui, uint, make_uint2)
DEFINE_WRITE_BUFFER3D(ui, uint)
#endif

#if defined(USE_3D) && defined(USE_LONG)
DEFINE_READ_BUFFER3D(l, long, make_long2)
DEFINE_WRITE_BUFFER3D(l, long)
#endif

#if defined(USE_3D) && defined(USE_ULONG)
DEFINE_READ_BUFFER3D(ul, ulong, make_ulong2)
DEFINE_WRITE_BUFFER3D(ul, ulong)
#endif

#if defined(USE_3D) && defined(USE_FLOAT)
DEFINE_READ_BUFFER3D(f, float, make_float2)
DEFINE_WRITE_BUFFER3D(f, float)
#endif

#if defined(USE_3D) && defined(USE_DOUBLE)
DEFINE_READ_BUFFER3D(d, double, make_double2)
DEFINE_WRITE_BUFFER3D(d, double)
#endif

#define DEFINE_READ_BUFFER2D(SUFFIX, TYPE, MAKE_FUNC) \
    __device__ inline TYPE##2 read_buffer2d ## SUFFIX(int read_buffer_width, int read_buffer_height, int read_buffer_depth, TYPE * buffer_var, int sampler, int2 position) { \
        int4 pos = make_int4(position.x, position.y, 0, 0); \
        pos.x = max((MINMAX_TYPE)pos.x, (MINMAX_TYPE)0); \
        pos.y = max((MINMAX_TYPE)pos.y, (MINMAX_TYPE)0); \
        pos.x = min((MINMAX_TYPE)pos.x, (MINMAX_TYPE)read_buffer_width - 1); \
        pos.y = min((MINMAX_TYPE)pos.y, (MINMAX_TYPE)read_buffer_height - 1); \
        int pos_in_buffer = pos.x + pos.y * read_buffer_width; \
        if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) return MAKE_FUNC(0, 0); \
        return MAKE_FUNC(buffer_var[pos_in_buffer], 0); \
    }

#define DEFINE_WRITE_BUFFER2D(SUFFIX, TYPE) \
    __device__ inline void write_buffer2d ## SUFFIX(int write_buffer_width, int write_buffer_height, int write_buffer_depth, TYPE * buffer_var, int2 pos, TYPE value) { \
        int pos_in_buffer = pos.x + pos.y * write_buffer_width; \
        if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) return; \
        buffer_var[pos_in_buffer] = value; \
    }

#if defined(USE_2D) && defined(USE_CHAR)
DEFINE_READ_BUFFER2D(c, char, make_char2)
DEFINE_WRITE_BUFFER2D(c, char)
#endif

#if defined(USE_2D) && defined(USE_UCHAR)
DEFINE_READ_BUFFER2D(uc, uchar, make_uchar2)
DEFINE_WRITE_BUFFER2D(uc, uchar)
#endif

#if defined(USE_2D) && defined(USE_SHORT)
DEFINE_READ_BUFFER2D(s, short, make_short2)
DEFINE_WRITE_BUFFER2D(s, short)
#endif

#if defined(USE_2D) && defined(USE_USHORT)
DEFINE_READ_BUFFER2D(us, ushort, make_ushort2)
DEFINE_WRITE_BUFFER2D(us, ushort)
#endif

#if defined(USE_2D) && defined(USE_INT)
DEFINE_READ_BUFFER2D(i, int, make_int2)
DEFINE_WRITE_BUFFER2D(i, int)
#endif

#if defined(USE_2D) && defined(USE_UINT)
DEFINE_READ_BUFFER2D(ui, uint, make_uint2)
DEFINE_WRITE_BUFFER2D(ui, uint)
#endif

#if defined(USE_2D) && defined(USE_LONG)
DEFINE_READ_BUFFER2D(l, long, make_long2)
DEFINE_WRITE_BUFFER2D(l, long)
#endif

#if defined(USE_2D) && defined(USE_ULONG)
DEFINE_READ_BUFFER2D(ul, ulong, make_ulong2)
DEFINE_WRITE_BUFFER2D(ul, ulong)
#endif

#if defined(USE_2D) && defined(USE_FLOAT)
DEFINE_READ_BUFFER2D(f, float, make_float2)
DEFINE_WRITE_BUFFER2D(f, float)
#endif

#if defined(USE_2D) && defined(USE_DOUBLE)
DEFINE_READ_BUFFER2D(d, double, make_double2)
DEFINE_WRITE_BUFFER2D(d, double)
#endif

#define DEFINE_READ_BUFFER1D(SUFFIX, TYPE, MAKE_FUNC) \
    __device__ inline TYPE##2 read_buffer1d ## SUFFIX(int read_buffer_width, int read_buffer_height, int read_buffer_depth, TYPE * buffer_var, int sampler, int position) { \
        int pos = max((MINMAX_TYPE)position, (MINMAX_TYPE)0); \
        pos = min((MINMAX_TYPE)pos, (MINMAX_TYPE)read_buffer_width - 1); \
        if (pos < 0 || pos >= read_buffer_width) return MAKE_FUNC(0, 0); \
        return MAKE_FUNC(buffer_var[pos], 0); \
    }

#define DEFINE_WRITE_BUFFER1D(SUFFIX, TYPE) \
    __device__ inline void write_buffer1d ## SUFFIX(int write_buffer_width, int write_buffer_height, int write_buffer_depth, TYPE * buffer_var, int pos, TYPE value) { \
        if (pos < 0 || pos >= write_buffer_width) return; \
        buffer_var[pos] = value; \
    }

#if defined(USE_1D) && defined(USE_CHAR)
DEFINE_READ_BUFFER1D(c, char, make_char2)
DEFINE_WRITE_BUFFER1D(c, char)
#endif

#if defined(USE_1D) && defined(USE_UCHAR)
DEFINE_READ_BUFFER1D(uc, uchar, make_uchar2)
DEFINE_WRITE_BUFFER1D(uc, uchar)
#endif

#if defined(USE_1D) && defined(USE_SHORT)
DEFINE_READ_BUFFER1D(s, short, make_short2)
DEFINE_WRITE_BUFFER1D(s, short)
#endif

#if defined(USE_1D) && defined(USE_USHORT)
DEFINE_READ_BUFFER1D(us, ushort, make_ushort2)
DEFINE_WRITE_BUFFER1D(us, ushort)
#endif

#if defined(USE_1D) && defined(USE_INT)
DEFINE_READ_BUFFER1D(i, int, make_int2)
DEFINE_WRITE_BUFFER1D(i, int)
#endif

#if defined(USE_1D) && defined(USE_UINT)
DEFINE_READ_BUFFER1D(ui, uint, make_uint2)
DEFINE_WRITE_BUFFER1D(ui, uint)
#endif

#if defined(USE_1D) && defined(USE_LONG)
DEFINE_READ_BUFFER1D(l, long, make_long2)
DEFINE_WRITE_BUFFER1D(l, long)
#endif

#if defined(USE_1D) && defined(USE_ULONG)
DEFINE_READ_BUFFER1D(ul, ulong, make_ulong2)
DEFINE_WRITE_BUFFER1D(ul, ulong)
#endif

#if defined(USE_1D) && defined(USE_FLOAT)
DEFINE_READ_BUFFER1D(f, float, make_float2)
DEFINE_WRITE_BUFFER1D(f, float)
#endif

#if defined(USE_1D) && defined(USE_DOUBLE)
DEFINE_READ_BUFFER1D(d, double, make_double2)
DEFINE_WRITE_BUFFER1D(d, double)
#endif

#endif

