// forward/backward Haar wavelet transforms
functions{
  // filter C coefficient vector
  vector filtC(vector C, int N) {
    vector[N] C1;
    for(n in 1:N) {
      C1[n] = (C[2 * n - 1] + C[2 * n]) / sqrt2();
    }
    return C1;
  }
  // filter D coefficient vector
  vector filtD(vector D, int N) {
    vector[N] D1;
    for(n in 1:N) {
      D1[n] = (D[2 * n - 1] - D[2 * n]) / sqrt2();
    }
    return D1;
  }
  // reconstruct C coefficient vector
  vector inv_filt(vector C, vector D, int N) {
    vector[2 * N] C1;
    for(n in 1:N) {
      C1[2 * n - 1] = (C[n] + D[n]) / sqrt2(); 
      C1[2 * n] = (C[n] - D[n])/ sqrt2();
    }
    return C1;
  }
  // forward Haar wavelet transform
  vector fwt(vector y) {
    int N = rows(y);
    int Ni = 0;
    vector[N] ywd;
    vector[N] C = y;
    while(N > 1) {
      N /= 2;
      ywd[(Ni + 1) : (Ni + N)] = filtD(C[1 : (2 * N)], N);
      C[1 : N] = filtC(C[1 : (2 * N)], N);
      Ni += N;
    }
    ywd[Ni + 1] = C[1];
    return ywd;
  }
  // inverse Haar wavelet transform
  vector iwt(vector ywd) {
    int N = rows(ywd);
    vector[N] y;
    int Nj = 1;
    y[1] = ywd[N];
    while(Nj < N) {
      y[1 : (2 * Nj)] = inv_filt(y[1 : Nj], ywd[(N - 2 * Nj + 1) : (N - Nj)], Nj); 
      Nj *= 2;
    }
    return y;
  }
}
data {
  int<lower=1> N;
  int<lower=2> K;
  vector[N] x;
  vector[N] y;
  matrix[N, N] W;
}
transformed data{
  vector[N] ywd = W' * y;
  real m0 = 0.01 * N;     // Expected number of large slopes
  real slab_scale = 3;    // Scale for large slopes
  real slab_scale2 = square(slab_scale);
  real slab_df = 25;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
}
parameters {
  vector[N] d_tilde;
  real<lower=0> sigma;
  vector<lower=0>[N] lambda;
  real<lower=0> c2_tilde;
  real<lower=0> tau_tilde;
}
transformed parameters {
  vector[M] beta;
  {
    real tau0 = (m0 / (M - m0)) * (sigma / sqrt(1.0 * N));
    real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)

    // c2 ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
    // Implies that marginally beta ~ student_t(slab_df, 0, slab_scale)
    real c2 = slab_scale2 * c2_tilde;

    vector[M] lambda_tilde =
      sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );

    // a ~ normal(0, tau * lambda_tilde)
    beta = tau * lambda_tilde .* beta_tilde;
  }
model {
  for(n in 1:N)
    target += log_mix(0.05, normal_lpdf(d[n] | 0, 10), normal_lpdf(d[n] | 0, 1E-4)); 
  sigma ~ exponential(1);
  ywd ~ normal(d, sigma);
}
generated quantities{
  vector[N] f = W * d;
}
