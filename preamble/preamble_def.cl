#ifndef PREAMBLE_DEFINE
#define PREAMBLE_DEFINE

#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_amd_printf : enable

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

inline uchar clij_convert_uchar_sat(float value) {
    if (value > 255) {
        return 255;
    }
    if (value < 0) {
        return 0;
    }
    return (uchar)value;
}

inline char clij_convert_char_sat(float value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return (char)value;
}

inline ushort clij_convert_ushort_sat(float value) {
    if (value > 65535) {
        return 65535;
    }
    if (value < 0) {
        return 0;
    }
    return (ushort)value;
}

inline short clij_convert_short_sat(float value) {
    if (value > 32767) {
        return 32767;
    }
    if (value < -32768) {
        return -32768;
    }
    return (short)value;
}

inline uint clij_convert_uint_sat(float value) {
    if (value > 4294967295) {
        return 4294967295;
    }
    if (value < 0) {
        return 0;
    }
    return (uint)value;
}

inline int clij_convert_int_sat(float value) {
    if (value > 2147483647) {
        return 2147483647;
    }
    if (value < -2147483648) {
        return -2147483648;
    }
    return (int)value;
}

inline uint clij_convert_ulong_sat(float value) {
    if (value > 18446744073709551615) {
        return 18446744073709551615;
    }
    if (value < 0) {
        return 0;
    }
    return (ulong)value;
}

inline int clij_convert_long_sat(float value) {
    if (value > 9223372036854775807) {
        return 9223372036854775807;
    }
    if (value < -9223372036854775808 ) {
        return -9223372036854775808 ;
    }
    return (long)value;
}

inline float clij_convert_float_sat(float value) {
    return value;
}

#define READ_IMAGE(a,b,c) READ_ ## a ## _IMAGE(a,b,c)
#define WRITE_IMAGE(a,b,c) WRITE_ ## a ## _IMAGE(a,b,c)

#endif // PREAMBLE_DEFINE