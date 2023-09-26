#ifndef PREAMBLE_DEFINE
#define PREAMBLE_DEFINE

#define sampler_t int

#define FLT_MIN 1.19209e-07
#define FLT_MAX 1e+37
#define MAX_ARRAY_SIZE 1000

#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int
#define ulong unsigned long

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
    if (value > 255) {
        return 255;
    }
    if (value < 0) {
        return 0;
    }
    return (uchar)value;
}

__device__ inline char clij_convert_char_sat(float value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return (char)value;
}

__device__ inline ushort clij_convert_ushort_sat(float value) {
    if (value > 65535) {
        return 65535;
    }
    if (value < 0) {
        return 0;
    }
    return (ushort)value;
}

__device__ inline short clij_convert_short_sat(float value) {
    if (value > 32767) {
        return 32767;
    }
    if (value < -32768) {
        return -32768;
    }
    return (short)value;
}

__device__ inline uint clij_convert_uint_sat(float value) {
    if (value > 4294967295) {
        return 4294967295;
    }
    if (value < 0) {
        return 0;
    }
    return (uint)value;
}

__device__ inline uint convert_uint_sat(float value) {
    if (value > 4294967295) {
        return 4294967295;
    }
    if (value < 0) {
        return 0;
    }
    return (uint)value;
}

__device__ inline int clij_convert_int_sat(float value) {
    if (value > 2147483647) {
        return 2147483647;
    }
    if (value < -2147483648) {
        return -2147483648;
    }
    return (int)value;
}

__device__ inline uint clij_convert_ulong_sat(float value) {
    if (value > 18446744073709551615) {
        return 18446744073709551615;
    }
    if (value < 0) {
        return 0;
    }
    return (ulong)value;
}

__device__ inline int clij_convert_long_sat(float value) {
    if (value > 9223372036854775807) {
        return 9223372036854775807;
    }
    if (value < -9223372036854775808 ) {
        return -9223372036854775808 ;
    }
    return (long)value;
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

#endif // PREAMBLE_DEFINE