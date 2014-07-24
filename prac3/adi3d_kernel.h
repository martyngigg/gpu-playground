
__global__ void GPU_adi_rhs(int NX, int NY, int NZ, float lam,
                            const float* __restrict__ d_u,
                                  float* __restrict__ d_du,
                                  float* __restrict__ d_ax,
                                  float* __restrict__ d_bx,
                                  float* __restrict__ d_cx,
                                  float* __restrict__ d_ay,
                                  float* __restrict__ d_by,
                                  float* __restrict__ d_cy,
                                  float* __restrict__ d_az,
                                  float* __restrict__ d_bz,
                                  float* __restrict__ d_cz)
{
  int   i, j, k, indg, active;
  float du, a, b, c;

  int NXM1 = NX-1;
  int NYM1 = NY-1;
  int NZM1 = NZ-1;

#define IOFF 1
#define JOFF NX
#define KOFF NX*NY

  //
  // set up indices for main block
  //

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;
  indg = i + j*NX;

  active = (i<NX) && (j<NY);

  //
  // loop over k-planes
  //

  for (k=0; k<NZ; k++) {

  //
  // calculate r.h.s. and set a,b,c, coefficients
  //

    if (active) {
      if (i==0 || i==NXM1 || j==0 || j==NYM1 || k==0 || k==NZM1) {
        du = 0.0f;          // Dirichlet b.c.'s
        a  = 0.0f;
        b  = 1.0f;
        c  = 0.0f;
      }
      else {
        du = lam * ( d_u[indg-IOFF] + d_u[indg+IOFF]
                   + d_u[indg-JOFF] + d_u[indg+JOFF]
                   + d_u[indg-KOFF] + d_u[indg+KOFF] - 6.0f*d_u[indg]); 
        a  = -0.5f*lam;
        b  =  1.0f + lam;
        c  = -0.5f*lam;
      }

      d_du[indg] = du;
      d_ax[indg] = a;
      d_bx[indg] = b;
      d_cx[indg] = c;
      d_ay[indg] = a;
      d_by[indg] = b;
      d_cy[indg] = c;
      d_az[indg] = a;
      d_bz[indg] = b;
      d_cz[indg] = c;

      indg += KOFF;
    }
  }
}


//
// tri-diagonal solve in x-direction
//

__global__ void GPU_adi_x(int NX, int NY, int NZ,
                          const float* __restrict__ d_a,
                          const float* __restrict__ d_b,
                          const float* __restrict__ d_c,
                                float* __restrict__ d_d)
{
  int   i, j, k, indg;
  float aa, bb, cc, dd, c[256], d[256];

  //
  // set up indices for main block
  //

  j    = threadIdx.x + blockIdx.x*blockDim.x;  // global indices
  k    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = NX*(j+k*NY);

  if ( (j<NY) && (k<NZ) ) {

  //
  // forward pass
  //

    bb   = 1.0f/d_b[indg];
    cc   = bb*d_c[indg];
    dd   = bb*d_d[indg];
    c[0] = cc;
    d[0] = dd;

    for (i=1; i<NX; i++) {
      indg = indg + 1;
      aa   = d_a[indg];
      bb   = d_b[indg] - aa*cc;
      dd   = d_d[indg] - aa*dd;
      bb   = 1.0f/bb;
      cc   = bb*d_c[indg];
      dd   = bb*dd;
      c[i] = cc;
      d[i] = dd;
    }

  //
  // reverse pass
  //

    d_d[indg] = dd;

    for (i=NX-2; i>=0; i--) {
      indg = indg - 1;
      dd = d[i] - c[i]*dd;
      d_d[indg] = dd;
    }
  }
}


//
// tri-diagonal solve in y-direction
//

__global__ void GPU_adi_y(int NX, int NY, int NZ,
                          const float* __restrict__ d_a,
                          const float* __restrict__ d_b,
                          const float* __restrict__ d_c,
                                float* __restrict__ d_d)
{
  int   i, j, k, indg;
  float aa, bb, cc, dd, c[256], d[256];

  //
  // set up indices for main block
  //

  i    = threadIdx.x + blockIdx.x*blockDim.x;  // global indices
  k    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = i + k*NX*NY;

  if ( (i<NX) && (k<NZ) ) {

  //
  // forward pass
  //

    bb   = 1.0f/d_b[indg];
    cc   = bb*d_c[indg];
    dd   = bb*d_d[indg];
    c[0] = cc;
    d[0] = dd;

    for (j=1; j<NY; j++) {
      indg = indg + NX;
      aa   = d_a[indg];
      bb   = d_b[indg] - aa*cc;
      dd   = d_d[indg] - aa*dd;
      bb   = 1.0f/bb;
      cc   = bb*d_c[indg];
      dd   = bb*dd;
      c[j] = cc;
      d[j] = dd;
    }

  //
  // reverse pass
  //

    d_d[indg] = dd;

    for (j=NY-2; j>=0; j--) {
      indg = indg - NX;
      dd = d[j] - c[j]*dd;
      d_d[indg] = dd;
    }

  }
}


//
// tri-diagonal solve in z-direction, and update solution
//

__global__ void GPU_adi_z(int NX, int NY, int NZ,
                                float* __restrict__ d_u,
                          const float* __restrict__ d_a,
                          const float* __restrict__ d_b,
                          const float* __restrict__ d_c,
                          const float* __restrict__ d_d)
{
  int   i, j, k, indg, off;
  float aa, bb, cc, dd, c[256], d[256];

  //
  // set up indices for main block
  //

  i    = threadIdx.x + blockIdx.x*blockDim.x;  // global indices
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = i+j*NX;
  off  = NX*NY;

  if ( (i<NX) && (j<NY) ) {

  //
  // forward pass
  //

    bb   = 1.0f/d_b[indg];
    cc   = bb*d_c[indg];
    dd   = bb*d_d[indg];
    c[0] = cc;
    d[0] = dd;

    for (k=1; k<NZ; k++) {
      indg = indg + off;
      aa   = d_a[indg];
      bb   = d_b[indg] - aa*cc;
      dd   = d_d[indg] - aa*dd;
      bb   = 1.0f/bb;
      cc   = bb*d_c[indg];
      dd   = bb*dd;
      c[k] = cc;
      d[k] = dd;
    }

  //
  // reverse pass
  //

    d_u[indg] += dd;

    for (k=NZ-2; k>=0; k--) {
      indg = indg - off;
      dd = d[k] - c[k]*dd;
      d_u[indg] += dd;
    }

  }
}

