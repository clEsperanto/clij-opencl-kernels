#define PRECISION ldexp(1.0f, -22)
#define DOUBLE_TYPE float // I did this because double isn't supported on Mac M1 @haesleinhuepf

// Returns 1 / sqrt(value)
inline DOUBLE_TYPE precise_rsqrt(DOUBLE_TYPE value) {
    // The opencl function rsqrt, might not give precise results.
    // This function uses the Newton method to improve the results precision.
    DOUBLE_TYPE x2 = value * 0.5;
    DOUBLE_TYPE y = rsqrt(value);
    y = y * ( 1.5 - ( x2 * y * y ) );   // Newton
    y = y * ( 1.5 - ( x2 * y * y ) );   // Newton
    y = y * ( 1.5 - ( x2 * y * y ) );   // Newton
    y = y * ( 1.5 - ( x2 * y * y ) );   // Newton
    y = y * ( 1.5 - ( x2 * y * y ) );   // Newton
    return y;
}

// Return the square root of value.
// This method has higher precision than opencl's sqrt() method.
inline DOUBLE_TYPE precise_sqrt(DOUBLE_TYPE value) {
    return value * precise_rsqrt(value);
}

inline void swap(DOUBLE_TYPE x[], int a, int b) {  // replace [] by * ? @strigaud
    DOUBLE_TYPE tmp = x[a];
    x[a] = x[b];
    x[b] = tmp;
}

// Calculates the two solutions of the equation: x^2 + c1 * x + c0 == 0
// The results are written to x[], smaller value first.
inline void solve_quadratic_equation(DOUBLE_TYPE c0, DOUBLE_TYPE c1, DOUBLE_TYPE x[]) {
    DOUBLE_TYPE p = 0.5 * c1;
    DOUBLE_TYPE dis = p * p - c0;
    dis = (dis > 0) ? precise_sqrt(dis) : 0;
    x[0] = (-p - dis);
    x[1] = (-p + dis);
}

// One iteration of Halleys method applied to the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
inline DOUBLE_TYPE halleys_method(DOUBLE_TYPE b0, DOUBLE_TYPE b1, DOUBLE_TYPE x) {
    DOUBLE_TYPE dy = 3 * x * x + b1;
    DOUBLE_TYPE y = (x * x + b1) * x + b0;	/* ...looks odd, but saves CPU time */
    DOUBLE_TYPE dx = y * dy / (dy * dy - 3 * y * x);
    return dx;
}

// Returns one solution to the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
inline DOUBLE_TYPE find_root(DOUBLE_TYPE b0, DOUBLE_TYPE b1) {
    if(b0 == 0)
        return 0;
    DOUBLE_TYPE w = max(fabs(b0), fabs(b1)) + 1.0; /* radius of root circle */
    DOUBLE_TYPE h = (b0 > 0.0) ? -w : w;
    DOUBLE_TYPE dx;
    do {					/* find 1st root by Halley's method */
        dx = halleys_method(b0, b1, h);
        h -= dx;
    } while (fabs(dx) > fabs(PRECISION * w));
    return h;
}

// Returns all three real solutions of the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
// The solutions are written to x[]. Smallest solution first.
inline void solve_cubic_scaled_equation(DOUBLE_TYPE b0, DOUBLE_TYPE b1, DOUBLE_TYPE x[]) {
    DOUBLE_TYPE h = find_root(b0, b1);
    x[2] = h;
    DOUBLE_TYPE c1 = h;			/* deflation; c2 is 1 */
    DOUBLE_TYPE c0 = c1 * h + b1;
    solve_quadratic_equation(c0, c1, x);
    if (x[1] > x[2]) {			/* sort results */
        swap(x, 1, 2);
        if (x[0] > x[1]) swap(x, 0, 1);
    }
}

inline int exponent_of(DOUBLE_TYPE f) {
    int exponent;
    frexp(f, &exponent);
    return exponent;
}

// Returns all three real solutions of the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
// The solutions are written to x[]. Smallest solution first.
inline void solve_depressed_cubic_equation(DOUBLE_TYPE b0, DOUBLE_TYPE b1, DOUBLE_TYPE x[]) {
    int e0 = exponent_of(b0) / 3;
    int e1 = exponent_of(b1) / 2;
    int e = - max(e0, e1);
    DOUBLE_TYPE scaleFactor = ldexp(1.0, -e);
    b1 = ldexp(b1, 2*e);
    b0 = ldexp(b0, 3*e);
    solve_cubic_scaled_equation(b0, b1, x);
    x[0] *= scaleFactor;
    x[1] *= scaleFactor;
    x[2] *= scaleFactor;
}

// Returns all three real solutions of the cubic equation:
//  x^3 + b2 * x^2 + b1 * x + b0 == 0
// The solutions are written to x[]. Smallest solution first.
inline void solve_cubic_equation(DOUBLE_TYPE b0, DOUBLE_TYPE b1, DOUBLE_TYPE b2, DOUBLE_TYPE x[]) {
    DOUBLE_TYPE s = 1.0 / 3.0 * b2;
    DOUBLE_TYPE q = (2. * s * s - b1) * s + b0;
    DOUBLE_TYPE p = b1 - b2 * s;
    solve_depressed_cubic_equation(q, p, x);
    x[0] = x[0] - s;
    x[1] = x[1] - s;
    x[2] = x[2] - s;
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/*
  This kernel computes the eigenvalues of the hessian matrix of a 3d image.

  Hessian matrix:
    [Ixx, Ixy, Ixz]
    [Ixy, Iyy, Iyz]
    [Ixz, Iyz, Izz]
  Where Ixx denotes the second derivative in x.

  Ixx and Iyy are calculated by convolving the image with the 1d kernel [1 -2 1].
  Ixy is calculated by a convolution with the 2d kernel:
    [ 0.25 0 -0.25]
    [    0 0     0]
    [-0.25 0  0.25]
*/
__kernel void hessian_eigenvalues(
    IMAGE_src_TYPE                src,
    IMAGE_small_eigenvalue_TYPE   small_eigenvalue,
    IMAGE_middle_eigenvalue_TYPE  middle_eigenvalue,
    IMAGE_large_eigenvalue_TYPE   large_eigenvalue
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const bool is_3d = GET_IMAGE_DEPTH(src) > 1;
  DOUBLE_TYPE eigenvalues[3] = {0, 0, 0};

  const DOUBLE_TYPE aab = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y - 1, z    , 0)).x; // 2d  eq. aa
  const DOUBLE_TYPE abb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y    , z    , 0)).x; // 2d  eq. ab
  const DOUBLE_TYPE acb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y + 1, z    , 0)).x; // 2d  eq. ac
  const DOUBLE_TYPE bab = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y - 1, z    , 0)).x; // 2d  eq. ba
  const DOUBLE_TYPE bbb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y    , z    , 0)).x; // 2d  eq. bb
  const DOUBLE_TYPE bcb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y + 1, z    , 0)).x; // 2d  eq. bc
  const DOUBLE_TYPE cab = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y - 1, z    , 0)).x; // 2d  eq. ca
  const DOUBLE_TYPE cbb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y    , z    , 0)).x; // 2d  eq. cb
  const DOUBLE_TYPE ccb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y + 1, z    , 0)).x; // 2d  eq. cc

  if (is_3d) { // missing computation for 3d
    const DOUBLE_TYPE aba = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y    , z - 1, 0)).x;
    const DOUBLE_TYPE abc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y    , z + 1, 0)).x;
    const DOUBLE_TYPE baa = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y - 1, z - 1, 0)).x;
    const DOUBLE_TYPE bac = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y - 1, z + 1, 0)).x;
    const DOUBLE_TYPE bba = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y    , z - 1, 0)).x; 
    const DOUBLE_TYPE bbc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y    , z + 1, 0)).x;
    const DOUBLE_TYPE bca = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y + 1, z - 1, 0)).x;
    const DOUBLE_TYPE bcc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y + 1, z + 1, 0)).x;
    const DOUBLE_TYPE cba = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y    , z - 1, 0)).x;
    const DOUBLE_TYPE cbc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y    , z + 1, 0)).x;
  }

  DOUBLE_TYPE g_xx = abb - 2 * bbb + cbb;
  DOUBLE_TYPE g_yy = bab - 2 * bbb + bcb;
  DOUBLE_TYPE g_zz = 0;
  DOUBLE_TYPE g_xy = (aab + ccb - acb - cab) * 0.25;
  if (is_3d) { // missing computation for 3d
    g_zz = bba - 2 * bbb + bbc;
    DOUBLE_TYPE g_xz = (aba + cbc - abc - cba) * 0.25;
    DOUBLE_TYPE g_yz = (baa + bcc - bac - bca) * 0.25;
  }
  
  const DOUBLE_TYPE a = (g_xx + g_yy + g_zz); // trace (g_zz is 0 if 2d_
  if (is_3d) { // missing computation for 3d
    const DOUBLE_TYPE b = g_xx * g_yy + g_xx * g_zz + g_yy * g_zz - g_xy * g_xy - g_xz * g_xz - g_yz * g_yz;
    const DOUBLE_TYPE c = g_xx * (g_yz * g_yz - g_yy * g_zz) + g_yy * g_xz * g_xz + g_zz * g_xy * g_xy - 2 * g_xy * g_xz * g_yz;
  }

  if(is_3d)
  {
    solve_cubic_equation(c, b, -a, eigenvalues);
  }
  else
  {
    eigenvalues[0] s = (DOUBLE_TYPE) (a / 2.0 - sqrt(4 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) / 2.0);
    eigenvalues[2] l = (DOUBLE_TYPE) (a / 2.0 + sqrt(4 * s_xy * s_xy + (s_xx - s_yy) * (s_xx - s_yy)) / 2.0);
  }

  WRITE_IMAGE(small_eigenvalue, sampler, POS_small_eigenvalue_INSTANCE(x, y, z), CONVERT_small_eigenvalue_PIXEL_TYPE(eigenvalues[0]));
  WRITE_IMAGE(middle_eigenvalue, sampler, POS_middle_eigenvalue_INSTANCE(x, y, z), CONVERT_middle_eigenvalue_PIXEL_TYPE(eigenvalues[1]));
  WRITE_IMAGE(large_eigenvalue, sampler, POS_large_eigenvalue_INSTANCE(x, y, z), CONVERT_large_eigenvalue_PIXEL_TYPE(eigenvalues[2]));
}
