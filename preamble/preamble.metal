// preamble.metal
#include <metal_stdlib>
using namespace metal;

// ---- Type aliases (matching OpenCL/CUDA naming) ----
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;

// ---- Sampler placeholder (unused in buffer path, kept for API parity) ----
typedef int sampler_t;

// ---- Constants ----
#ifndef M_PI
    #define M_PI     3.14159265358979323846f
#endif
#ifndef M_LOG2E
    #define M_LOG2E  1.4426950408889634074f
#endif
#ifndef M_LOG10E
    #define M_LOG10E 0.43429448190325182765f
#endif
#ifndef M_LN2
    #define M_LN2    0.69314718055994530942f
#endif
#ifndef M_LN10
    #define M_LN10   2.30258509299404568402f
#endif

#ifndef FLT_MIN
    #define FLT_MIN 1.175494e-38f
#endif
#ifndef FLT_MAX
    #define FLT_MAX 3.402823e+38f
#endif

// ---- Function aliases and helpers ----
#define mad fma
#define cbrt(x) __cle_cbrt(x)
#define pow(b,e) __cle_pow(b,e)
#define frexp(x, p) metal::frexp(x, *(p))

#define convert_uchar_sat clij_convert_uchar_sat
#define convert_char_sat clij_convert_char_sat
#define convert_ushort_sat clij_convert_ushort_sat
#define convert_short_sat clij_convert_short_sat
#define convert_uint_sat clij_convert_uint_sat
#define convert_int_sat clij_convert_int_sat
#define convert_ulong_sat clij_convert_ulong_sat
#define convert_long_sat clij_convert_long_sat
#define convert_float_sat clij_convert_float_sat


#define MAX_ARRAY_SIZE 1000

// ---- Sampler flag constants (for API parity) ----
#define CLK_NORMALIZED_COORDS_FALSE 1
#define CLK_ADDRESS_CLAMP_TO_EDGE   2
#define CLK_FILTER_NEAREST          4
#define CLK_NORMALIZED_COORDS_TRUE  8
#define CLK_ADDRESS_CLAMP           16
#define CLK_FILTER_LINEAR           32
#define CLK_ADDRESS_NONE            64

// ---- Vector helpers ----
// Metal has native int2, int4, float2, float4, etc. via <metal_stdlib>
// make_*2 / make_*4 equivalents:
inline int2   make_int2(int x, int y)               { return int2(x, y); }
inline int4   make_int4(int x, int y, int z, int w) { return int4(x, y, z, w); }
inline float2 make_float2(float x, float y)         { return float2(x, y); }
inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
inline char2  make_char2(char x, char y)            { return char2(x, y); }
inline uchar2 make_uchar2(uchar x, uchar y)         { return uchar2(x, y); }
inline short2 make_short2(short x, short y)         { return short2(x, y); }
inline ushort2 make_ushort2(ushort x, ushort y)     { return ushort2(x, y); }
inline uint2  make_uint2(uint x, uint y)            { return uint2(x, y); }
inline long2  make_long2(long x, long y)            { return long2(x, y); }
inline ulong2 make_ulong2(ulong x, ulong y)         { return ulong2(x, y); }

// ---- Math helpers ----
inline float4 metal_cross(float4 a, float4 b) {
    return float4(a.y*b.z - a.z*b.y,
                  a.z*b.x - a.x*b.z,
                  a.x*b.y - a.y*b.x,
                  0.0f);
}

inline float metal_dot4(float4 a, float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline float metal_length4(float4 v) {
    return sqrt(metal_dot4(v, v));
}

inline float __cle_cbrt(float x) {
  return x < 0.0f ? -metal::pow(-x, 1.0f / 3.0f) : metal::pow(x, 1.0f / 3.0f);
}

inline float __cle_pow(float x, float y) {
  float yi = metal::rint(y);
  if (metal::fabs(y - yi) < 1e-6f) {
    int n = (int)yi;
    float base = x;
    if (n < 0) { n = -n; base = 1.0f / base; }
    float result = 1.0f;
    for (int i = 0; i < n; ++i) result *= base;
    return result;
  }
  return metal::pow(x, y);
}

// ---- Atomic add (device address space) ----
inline uint metal_atomic_add(volatile device atomic_uint* address, uint value) {
    return atomic_fetch_add_explicit(address, value, memory_order_relaxed);
}

// ---- Saturating type converters ----
inline uchar  clij_convert_uchar_sat(float v)  { return (uchar)  clamp(v, 0.0f, 255.0f); }
inline char   clij_convert_char_sat(float v)   { return (char)   clamp(v, -128.0f, 127.0f); }
inline ushort clij_convert_ushort_sat(float v) { return (ushort) clamp(v, 0.0f, 65535.0f); }
inline short  clij_convert_short_sat(float v)  { return (short)  clamp(v, -32768.0f, 32767.0f); }
inline uint   clij_convert_uint_sat(float v)   { return (uint)   clamp(v, 0.0f, 4294967295.0f); }
inline int    clij_convert_int_sat(float v)    { return (int)    clamp(v, -2147483648.0f, 2147483647.0f); }
inline ulong  clij_convert_ulong_sat(float v)  { return (ulong)  clamp(v, 0.0f, 18446744073709551615.0f); }
inline long   clij_convert_long_sat(float v)   { return (long)   clamp(v, -9223372036854775808.0f, 9223372036854775807.0f); }
inline float  clij_convert_float_sat(float v)  { return v; }

// ---- READ/WRITE IMAGE macros (for API parity) ----
#define READ_IMAGE(a,b,c)  READ_ ## a ## _IMAGE(a,b,c)
#define WRITE_IMAGE(a,b,c) WRITE_ ## a ## _IMAGE(a,b,c)

// ---- Buffer read/write macros ----
// Metal uses 'device' address space instead of __global.
// Buffer pointers are passed as `device TYPE*`.

#ifndef BUFFER_READ_WRITE
#define BUFFER_READ_WRITE 1

#define DEFINE_READ_BUFFER3D(SUFFIX, TYPE, MAKE_FUNC) \
    inline TYPE##2 read_buffer3d ## SUFFIX(int w, int h, int d, device TYPE* buf, int sampler, int4 position) { \
        int px = clamp(position.x, 0, w - 1); \
        int py = clamp(position.y, 0, h - 1); \
        int pz = clamp(position.z, 0, d - 1); \
        if (px < 0 || px >= w || py < 0 || py >= h || pz < 0 || pz >= d) return MAKE_FUNC(0, 0); \
        int idx = px + py * w + pz * w * h; \
        return MAKE_FUNC(buf[idx], 0); \
    }

#define DEFINE_WRITE_BUFFER3D(SUFFIX, TYPE) \
    inline void write_buffer3d ## SUFFIX(int w, int h, int d, device TYPE* buf, int4 pos, TYPE value) { \
        if (pos.x < 0 || pos.x >= w || pos.y < 0 || pos.y >= h || pos.z < 0 || pos.z >= d) return; \
        buf[pos.x + pos.y * w + pos.z * w * h] = value; \
    }

#define DEFINE_READ_BUFFER2D(SUFFIX, TYPE, MAKE_FUNC) \
    inline TYPE##2 read_buffer2d ## SUFFIX(int w, int h, int d, device TYPE* buf, int sampler, int2 position) { \
        int px = clamp(position.x, 0, w - 1); \
        int py = clamp(position.y, 0, h - 1); \
        if (px < 0 || px >= w || py < 0 || py >= h) return MAKE_FUNC(0, 0); \
        return MAKE_FUNC(buf[px + py * w], 0); \
    }

#define DEFINE_WRITE_BUFFER2D(SUFFIX, TYPE) \
    inline void write_buffer2d ## SUFFIX(int w, int h, int d, device TYPE* buf, int2 pos, TYPE value) { \
        if (pos.x < 0 || pos.x >= w || pos.y < 0 || pos.y >= h) return; \
        buf[pos.x + pos.y * w] = value; \
    }

#define DEFINE_READ_BUFFER1D(SUFFIX, TYPE, MAKE_FUNC) \
    inline TYPE##2 read_buffer1d ## SUFFIX(int w, int h, int d, device TYPE* buf, int sampler, int position) { \
        int pos = clamp(position, 0, w - 1); \
        if (pos < 0 || pos >= w) return MAKE_FUNC(0, 0); \
        return MAKE_FUNC(buf[pos], 0); \
    }

#define DEFINE_WRITE_BUFFER1D(SUFFIX, TYPE) \
    inline void write_buffer1d ## SUFFIX(int w, int h, int d, device TYPE* buf, int pos, TYPE value) { \
        if (pos < 0 || pos >= w) return; \
        buf[pos] = value; \
    }

// ---- 3D instantiations ----
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

// ---- 2D instantiations ----
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

// ---- 1D instantiations ----
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

#endif // BUFFER_READ_WRITE