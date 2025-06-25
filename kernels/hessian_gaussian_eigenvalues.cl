#define PRECISION ldexp(1.0f, -22)
#define DOUBLE_TYPE float

// Returns 1 / sqrt(value)
inline DOUBLE_TYPE precise_rsqrt(DOUBLE_TYPE value) {
  // The opencl function rsqrt, might not give precise results.
  // This function uses the Newton method to improve the results precision.
  DOUBLE_TYPE x2 = value * 0.5;
  DOUBLE_TYPE y = rsqrt(value);
  y = y * (1.5 - (x2 * y * y)); // Newton
  y = y * (1.5 - (x2 * y * y)); // Newton
  y = y * (1.5 - (x2 * y * y)); // Newton
  y = y * (1.5 - (x2 * y * y)); // Newton
  y = y * (1.5 - (x2 * y * y)); // Newton
  return y;
}

// Return the square root of value.
// This method has higher precision than opencl's sqrt() method.
inline DOUBLE_TYPE precise_sqrt(DOUBLE_TYPE value) {
  return value * precise_rsqrt(value);
}

inline void swap(DOUBLE_TYPE x[], int a, int b) { // replace [] by * ? @strigaud
  DOUBLE_TYPE tmp = x[a];
  x[a] = x[b];
  x[b] = tmp;
}

// Calculates the two solutions of the equation: x^2 + c1 * x + c0 == 0
// The results are written to x[], smaller value first.
inline void solve_quadratic_equation(DOUBLE_TYPE c0, DOUBLE_TYPE c1,
                                     DOUBLE_TYPE x[]) {
  DOUBLE_TYPE p = 0.5 * c1;
  DOUBLE_TYPE dis = p * p - c0;
  dis = (dis > 0) ? precise_sqrt(dis) : 0;
  x[0] = (-p - dis);
  x[1] = (-p + dis);
}

// One iteration of Halleys method applied to the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
inline DOUBLE_TYPE halleys_method(DOUBLE_TYPE b0, DOUBLE_TYPE b1,
                                  DOUBLE_TYPE x) {
  DOUBLE_TYPE dy = 3 * x * x + b1;
  DOUBLE_TYPE y = (x * x + b1) * x + b0; /* ...looks odd, but saves CPU time */
  DOUBLE_TYPE dx = y * dy / (dy * dy - 3 * y * x);
  return dx;
}

// Returns one solution to the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
inline DOUBLE_TYPE find_root(DOUBLE_TYPE b0, DOUBLE_TYPE b1) {
  if (b0 == 0)
    return 0;
  DOUBLE_TYPE w = max(fabs(b0), fabs(b1)) + 1.0; /* radius of root circle */
  DOUBLE_TYPE h = (b0 > 0.0) ? -w : w;
  DOUBLE_TYPE dx;
  do { /* find 1st root by Halley's method */
    dx = halleys_method(b0, b1, h);
    h -= dx;
  } while (fabs(dx) > fabs(PRECISION * w));
  return h;
}

// Returns all three real solutions of the depressed cubic equation:
//  x^3 + b1 * x + b0 == 0
// The solutions are written to x[]. Smallest solution first.
inline void solve_cubic_scaled_equation(DOUBLE_TYPE b0, DOUBLE_TYPE b1,
                                        DOUBLE_TYPE x[]) {
  DOUBLE_TYPE h = find_root(b0, b1);
  x[2] = h;
  DOUBLE_TYPE c1 = h; /* deflation; c2 is 1 */
  DOUBLE_TYPE c0 = c1 * h + b1;
  solve_quadratic_equation(c0, c1, x);
  if (x[1] > x[2]) { /* sort results */
    swap(x, 1, 2);
    if (x[0] > x[1])
      swap(x, 0, 1);
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
inline void solve_depressed_cubic_equation(DOUBLE_TYPE b0, DOUBLE_TYPE b1,
                                           DOUBLE_TYPE x[]) {
  int e0 = exponent_of(b0) / 3;
  int e1 = exponent_of(b1) / 2;
  int e = -max(e0, e1);
  DOUBLE_TYPE scaleFactor = ldexp(1.0, -e);
  b1 = ldexp(b1, 2 * e);
  b0 = ldexp(b0, 3 * e);
  solve_cubic_scaled_equation(b0, b1, x);
  x[0] *= scaleFactor;
  x[1] *= scaleFactor;
  x[2] *= scaleFactor;
}

// Returns all three real solutions of the cubic equation:
//  x^3 + b2 * x^2 + b1 * x + b0 == 0
// The solutions are written to x[]. Smallest solution first.
inline void solve_cubic_equation(DOUBLE_TYPE b0, DOUBLE_TYPE b1, DOUBLE_TYPE b2,
                                 DOUBLE_TYPE x[]) {
  DOUBLE_TYPE s = 1.0 / 3.0 * b2;
  DOUBLE_TYPE q = (2. * s * s - b1) * s + b0;
  DOUBLE_TYPE p = b1 - b2 * s;
  solve_depressed_cubic_equation(q, p, x);
  x[0] = x[0] - s;
  x[1] = x[1] - s;
  x[2] = x[2] - s;
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline void compute_gaussian_hessian_2d(
    IMAGE_src_TYPE src, // Input 2D image
    IMAGE_gfd_TYPE gfd, // Gaussian first derivative 1d array
    IMAGE_gsd_TYPE gsd, // Gaussian second derivative 1d array
    int x, int y,       // Coordinates in the image
    DOUBLE_TYPE hessian[]) {

  // Temporary variables for derivatives
  float deriv_xx = 0.0f;
  float deriv_yy = 0.0f;
  float deriv_xy = 0.0f;

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int kernel_size = GET_IMAGE_WIDTH(gfd);
  const int half_kernel = kernel_size / 2;

  // Compute second derivatives along x and y (Ixx, Iyy)
  for (int i = 0; i < kernel_size; i++) {
    int offset_x = x + (i - half_kernel);
    int offset_y = y + (i - half_kernel);

    float value_x =
        (float)READ_IMAGE(src, sampler, POS_src_INSTANCE(offset_x, y, 0, 0)).x;
    float value_y =
        (float)READ_IMAGE(src, sampler, POS_src_INSTANCE(x, offset_y, 0, 0)).x;

    deriv_xx += value_x * gsd[i];
    deriv_yy += value_y * gsd[i];
  }

  // Compute mixed derivative (Ixy)
  for (int i = 0; i < kernel_size; i++) {
    int offset_x = x + (i - half_kernel);
    for (int j = 0; j < kernel_size; j++) {
      int offset_y = y + (j - half_kernel);

      float value_xy =
          (float)READ_IMAGE(src, sampler,
                            POS_src_INSTANCE(offset_x, offset_y, 0, 0))
              .x;

      deriv_xy += value_xy * gfd[i] * gfd[j];
    }
  }

  // Store results in the Hessian matrix
  hessian[0] = deriv_xx; // xx
  hessian[1] = deriv_xy; // xy
  hessian[3] = deriv_yy; // yy
}

inline void compute_gaussian_hessian_3d(
    IMAGE_src_TYPE src,  // Input 2D image
    IMAGE_gfd_TYPE gfd,  // Gaussian first derivative 1d array (normalized)
    IMAGE_gsd_TYPE gsd,  // Gaussian second derivative 1d array (normalized)
    int x, int y, int z, // Coordinates in the image
    DOUBLE_TYPE hessian[]) {

  // Temporary variables for derivatives
  float deriv_xx = 0.0f;
  float deriv_yy = 0.0f;
  float deriv_zz = 0.0f;
  float deriv_xy = 0.0f;
  float deriv_xz = 0.0f;
  float deriv_yz = 0.0f;

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);
  const int kernel_size = GET_IMAGE_WIDTH(gfd);
  const int half_kernel = kernel_size / 2;

  // Compute second derivatives along x, y, z (Ixx, Iyy, Izz)
  for (int i = 0; i < kernel_size; i++) {
    int offset_x = x + (i - half_kernel);
    int offset_y = y + (i - half_kernel);
    int offset_z = z + (i - half_kernel);

    float value_x =
        (float)READ_IMAGE(src, sampler, POS_src_INSTANCE(offset_x, y, z, 0)).x;
    float value_y =
        (float)READ_IMAGE(src, sampler, POS_src_INSTANCE(x, offset_y, z, 0)).x;
    float value_z =
        (float)READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, offset_z, 0)).x;

    deriv_xx += value_x * gsd[i];
    deriv_yy += value_y * gsd[i];
    deriv_zz += value_z * gsd[i];
  }

  // Compute mixed derivatives (Ixy, Ixz, Iyz)
  for (int i = 0; i < kernel_size; i++) {
    int offset_x = x + (i - half_kernel);

    for (int j = 0; j < kernel_size; j++) {
      int offset_y_j = y + (j - half_kernel);
      int offset_z_j = z + (j - half_kernel);

      float value_xy =
          (float)READ_IMAGE(src, sampler,
                            POS_src_INSTANCE(offset_x, offset_y_j, z, 0))
              .x;
      float value_xz =
          (float)READ_IMAGE(src, sampler,
                            POS_src_INSTANCE(offset_x, y, offset_z_j, 0))
              .x;
      float value_yz =
          (float)READ_IMAGE(src, sampler,
                            POS_src_INSTANCE(x, offset_y_j, offset_z_j, 0))
              .x;

      deriv_xy += value_xy * gfd[i] * gfd[j];
      deriv_xz += value_xz * gfd[i] * gfd[j];
      deriv_yz += value_yz * gfd[i] * gfd[j];
    }
  }

  // Store results in the Hessian matrix
  hessian[0] = deriv_xx; // xx
  hessian[1] = deriv_xy; // xy
  hessian[2] = deriv_xz; // xz
  hessian[3] = deriv_yy; // yy
  hessian[4] = deriv_yz; // yz
  hessian[5] = deriv_zz; // zz
}

/*
  This kernel computes the eigenvalues of the hessian matrix of a 3d image using
  the Gaussian derivative.

  Hessian matrix:
    [Ixx, Ixy, Ixz]
    [Ixy, Iyy, Iyz]
    [Ixz, Iyz, Izz]
  Where Ixx denotes the second derivative in x.

  Ixx and Iyy are calculated by convolving the image with the 1d kernel [1 -2
  1]. Ixy is calculated by a convolution with the 2d kernel: [ 0.25 0 -0.25] [
  0 0     0]
    [-0.25 0  0.25]
*/
__kernel void hessian_gaussian_eigenvalues(
    IMAGE_src_TYPE src, // Input 2D image
    IMAGE_gfd_TYPE gfd, // Gaussian first derivative 1d array
    IMAGE_gsd_TYPE gsd, // Gaussian second derivative 1d array
    IMAGE_small_eigenvalue_TYPE small_eigenvalue,
    IMAGE_middle_eigenvalue_TYPE middle_eigenvalue,
    IMAGE_large_eigenvalue_TYPE large_eigenvalue) {

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const bool is_3d = GET_IMAGE_DEPTH(src) > 1;
  DOUBLE_TYPE eigenvalues[3] = {0, 0, 0};
  DOUBLE_TYPE hessian[6] = {0, 0, 0, 0, 0, 0};

  if (is_3d) {
    compute_gaussian_hessian_3d(src, gfd, gsd, x, y, z, hessian);
  } else {
    compute_gaussian_hessian_2d(src, gfd, gsd, x, y, hessian);
  }

  DOUBLE_TYPE a, b, c;
  a = (hessian[0] + hessian[3] + hessian[5]); // trace
  if (is_3d) {
    b = hessian[0] * hessian[3] + hessian[0] * hessian[5] +
        hessian[3] * hessian[5] - hessian[1] * hessian[1] -
        hessian[2] * hessian[2] - hessian[4] * hessian[4];
    c = hessian[0] * (hessian[4] * hessian[4] - hessian[3] * hessian[5]) +
        hessian[3] * hessian[2] * hessian[2] +
        hessian[5] * hessian[1] * hessian[1] -
        2 * hessian[1] * hessian[2] * hessian[4];
    solve_cubic_equation(c, b, -a, eigenvalues);
    WRITE_IMAGE(middle_eigenvalue, POS_middle_eigenvalue_INSTANCE(x, y, z, 0),
                CONVERT_middle_eigenvalue_PIXEL_TYPE(eigenvalues[1]));
  } else {
    eigenvalues[0] =
        (DOUBLE_TYPE)(a / 2.0 - sqrt(4 * hessian[1] * hessian[1] +
                                     (hessian[0] - hessian[3]) *
                                         (hessian[0] - hessian[3])) /
                                    2.0);
    eigenvalues[2] =
        (DOUBLE_TYPE)(a / 2.0 + sqrt(4 * hessian[1] * hessian[1] +
                                     (hessian[0] - hessian[3]) *
                                         (hessian[0] - hessian[3])) /
                                    2.0);
  }
  WRITE_IMAGE(small_eigenvalue, POS_small_eigenvalue_INSTANCE(x, y, z, 0),
              CONVERT_small_eigenvalue_PIXEL_TYPE(eigenvalues[0]));
  WRITE_IMAGE(large_eigenvalue, POS_large_eigenvalue_INSTANCE(x, y, z, 0),
              CONVERT_large_eigenvalue_PIXEL_TYPE(eigenvalues[2]));
}
