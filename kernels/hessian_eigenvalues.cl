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

inline void compute_hessian_2D(IMAGE_src_TYPE src, const int x, const int y, DOUBLE_TYPE hessian[]){
    DOUBLE_TYPE a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x+1, y+1, z, 0)).x;
    DOUBLE_TYPE b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x+1, y-1, z, 0)).x;
    DOUBLE_TYPE c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y+1, z, 0)).x;
    DOUBLE_TYPE d = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y-1, z, 0)).x;
    const DOUBLE_TYPE dxy = (a - b - c + d)* 0.25;
    hessian[1] = (a - b - c + d)/4; //xy
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y, z, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y, z, 0)).x;
    hessian[0] = a - 2 * b + c;  //xx
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y+1, z, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y-1, z, 0)).x;
    hessian[3] = a - 2 * b + c;  //yy
    // matrix [xx, xy, 0, yy, 0, zz]
}

inline void compute_hessian_3D(IMAGE_src_TYPE src, const int x, const int y, const int z, DOUBLE_TYPE hessian[]){
    DOUBLE_TYPE a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x+1, y+1, z, 0)).x;
    DOUBLE_TYPE b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x+1, y-1, z, 0)).x;
    DOUBLE_TYPE c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y+1, z, 0)).x;
    DOUBLE_TYPE d = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y-1, z, 0)).x;
    hessian[1]  = (a - b - c + d)* 0.25; //xy
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x+1, y, z+1, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x+1, y, z-1, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y, z+1, 0)).x;
    d = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y, z-1, 0)).x;
    hessian[2] = (a - b - c + d)* 0.25; //xz
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y+1, z+1, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y+1, z-1, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y-1, z+1, 0)).x;
    d = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y-1, z-1, 0)).x;
    hessian[4]  = (a - b - c + d)* 0.25; //yz
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y, z, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x-1, y, z, 0)).x;
    hessian[0] = a - 2 * b + c; //xx
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y+1, z, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y-1, z, 0)).x;
    hessian[3] = a - 2 * b + c; //yy
    a = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z+1, 0)).x;
    b = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    c = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z-1, 0)).x;
    hessian[5] = a - 2 * b + c; //zz
    // hessian[0] xx 
    // hessian[1] xy
    // hessian[2] xz
    // hessian[3] yy
    // hessian[4] yz
    // hessian[5] zz
}

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
  DOUBLE_TYPE hessian[6] = {0, 0, 0, 0, 0, 0};

  if (is_3d) {
  compute_hessian_3D(src, x, y, z, hessian);
  }
  else{
  compute_hessian_2D(src, x, y, hessian);
  }
    
  DOUBLE_TYPE a, b, c;
  a = (hessian[0] + hessian[3] + hessian[5]);
  if (is_3d) { // missing computation for 3d
    b = hessian[0] * hessian[3] + hessian[0] * hessian[5] + hessian[3] * hessian[5] - hessian[1] * hessian[1] - hessian[2] * hessian[2] - hessian[4] * hessian[4];
    c = hessian[0] * (hessian[4] * hessian[4] - hessian[3] * hessian[5]) + hessian[3] * hessian[2] * hessian[2] + hessian[5] * hessian[1] * hessian[1] - 2 * hessian[1] * hessian[2] * hessian[4];
    solve_cubic_equation(c, b, -a, eigenvalues);
  }
  else
  {
    eigenvalues[0] = (DOUBLE_TYPE) (a / 2.0 - sqrt(4 * hessian[1] * hessian[1] + (hessian[0] - hessian[3]) * (hessian[0] - hessian[3])) / 2.0);
    eigenvalues[2] = (DOUBLE_TYPE) (a / 2.0 + sqrt(4 * hessian[1] * hessian[1] + (hessian[0] - hessian[3]) * (hessian[0] - hessian[3])) / 2.0);
  }

  WRITE_IMAGE(small_eigenvalue, POS_small_eigenvalue_INSTANCE(x, y, z, 0), CONVERT_small_eigenvalue_PIXEL_TYPE(eigenvalues[0]));
  WRITE_IMAGE(middle_eigenvalue, POS_middle_eigenvalue_INSTANCE(x, y, z, 0), CONVERT_middle_eigenvalue_PIXEL_TYPE(eigenvalues[1]));
  WRITE_IMAGE(large_eigenvalue, POS_large_eigenvalue_INSTANCE(x, y, z, 0), CONVERT_large_eigenvalue_PIXEL_TYPE(eigenvalues[2]));
}


// __kernel void hessian_eigenvalues(
//     IMAGE_src_TYPE                src,
//     IMAGE_small_eigenvalue_TYPE   small_eigenvalue,
//     IMAGE_middle_eigenvalue_TYPE  middle_eigenvalue,
//     IMAGE_large_eigenvalue_TYPE   large_eigenvalue
// )
// {
//   const int x = get_global_id(0);
//   const int y = get_global_id(1);
//   const int z = get_global_id(2);

//   const bool is_3d = GET_IMAGE_DEPTH(src) > 1;
//   DOUBLE_TYPE eigenvalues[3] = {0, 0, 0};
//   DOUBLE_TYPE aab = 0;
//   DOUBLE_TYPE abb = 0;
//   DOUBLE_TYPE acb = 0;
//   DOUBLE_TYPE bab = 0;
//   DOUBLE_TYPE bbb = 0;
//   DOUBLE_TYPE bcb = 0;
//   DOUBLE_TYPE cab = 0;
//   DOUBLE_TYPE cbb = 0;
//   DOUBLE_TYPE ccb = 0;
//   DOUBLE_TYPE aba = 0;
//   DOUBLE_TYPE abc = 0;
//   DOUBLE_TYPE baa = 0;
//   DOUBLE_TYPE bac = 0;
//   DOUBLE_TYPE bba = 0;
//   DOUBLE_TYPE bbc = 0;
//   DOUBLE_TYPE bca = 0;
//   DOUBLE_TYPE bcc = 0;
//   DOUBLE_TYPE cba = 0;
//   DOUBLE_TYPE cbc = 0;

//   aab = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y - 1, z    , 0)).x;; // 2d  eq. aa
//   abb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y    , z    , 0)).x;; // 2d  eq. ab
//   acb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y + 1, z    , 0)).x;; // 2d  eq. ac
//   bab = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y - 1, z    , 0)).x;; // 2d  eq. ba
//   bbb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y    , z    , 0)).x;; // 2d  eq. bb
//   bcb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y + 1, z    , 0)).x;; // 2d  eq. bc
//   cab = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y - 1, z    , 0)).x;; // 2d  eq. ca
//   cbb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y    , z    , 0)).x;; // 2d  eq. cb
//   ccb = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y + 1, z    , 0)).x;; // 2d  eq. cc

//   if (is_3d) { // missing computation for 3d
//     aba = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y    , z - 1, 0)).x;;
//     abc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y    , z + 1, 0)).x;;
//     baa = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y - 1, z - 1, 0)).x;;
//     bac = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y - 1, z + 1, 0)).x;;
//     bba = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y    , z - 1, 0)).x;; 
//     bbc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y    , z + 1, 0)).x;;
//     bca = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y + 1, z - 1, 0)).x;;
//     bcc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x    , y + 1, z + 1, 0)).x;;
//     cba = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y    , z - 1, 0)).x;;
//     cbc = (DOUBLE_TYPE) READ_src_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y    , z + 1, 0)).x;;
//   }

// DOUBLE_TYPE g_xx = 0;
// DOUBLE_TYPE g_yy = 0;
// DOUBLE_TYPE g_zz = 0;
// DOUBLE_TYPE g_xy = 0;
// DOUBLE_TYPE g_xz = 0;
// DOUBLE_TYPE g_yz = 0;

// DOUBLE_TYPE a = 0;
// DOUBLE_TYPE b = 0;
// DOUBLE_TYPE c = 0;

//   g_xx = abb - 2 * bbb + cbb;
//   g_yy = bab - 2 * bbb + bcb;
//   g_zz = 0;
//   g_xy = (aab + ccb - acb - cab) * 0.25;
//   if (is_3d) { // missing computation for 3d
//     g_zz = bba - 2 * bbb + bbc;
//     g_xz = (aba + cbc - abc - cba) * 0.25;
//     g_yz = (baa + bcc - bac - bca) * 0.25;
//   }
  
//   a = (g_xx + g_yy + g_zz); // trace (g_zz is 0 if 2d_
//   if (is_3d) { // missing computation for 3d
//     b = g_xx * g_yy + g_xx * g_zz + g_yy * g_zz - g_xy * g_xy - g_xz * g_xz - g_yz * g_yz;
//     c = g_xx * (g_yz * g_yz - g_yy * g_zz) + g_yy * g_xz * g_xz + g_zz * g_xy * g_xy - 2 * g_xy * g_xz * g_yz;
//   }

//   if(is_3d)
//   {
//     solve_cubic_equation(c, b, -a, eigenvalues);
//   }
//   else
//   {
//     eigenvalues[0] = (DOUBLE_TYPE) (a / 2.0 - sqrt(4 * g_xy * g_xy + (g_xx - g_yy) * (g_xx - g_yy)) / 2.0);
//     eigenvalues[2] = (DOUBLE_TYPE) (a / 2.0 + sqrt(4 * g_xy * g_xy + (g_xx - g_yy) * (g_xx - g_yy)) / 2.0);
//   }

//   WRITE_IMAGE(small_eigenvalue, POS_small_eigenvalue_INSTANCE(x, y, z, 0), CONVERT_small_eigenvalue_PIXEL_TYPE(eigenvalues[0]));
//   WRITE_IMAGE(middle_eigenvalue, POS_middle_eigenvalue_INSTANCE(x, y, z, 0), CONVERT_middle_eigenvalue_PIXEL_TYPE(eigenvalues[1]));
//   WRITE_IMAGE(large_eigenvalue, POS_large_eigenvalue_INSTANCE(x, y, z, 0), CONVERT_large_eigenvalue_PIXEL_TYPE(eigenvalues[2]));
// }
