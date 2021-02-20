data {
  // dimensions
  int<lower=1> N;
  int<lower=1, upper=N> N_y;
  int<lower=1> p;
  int<lower=1> q;
  // covariates
  matrix[N_y, p] X;
  matrix[N, q] Z;
  // responses
  int<lower=0, upper=1> D[N];
  vector[N_y] y;
}
parameters {
  vector[p] beta;
  vector[q] gamma;
  real<lower=-1,upper=1> rho;
  real<lower=0> sigma_e;
}
model {
  // naive (truncated) priors
  beta ~ normal(0, 1);
  gamma ~ normal(0, 1);
  rho ~ normal(0, 0.25);
  sigma_e ~ normal(0, 1);
  {
    // log-likelihood
    vector[N_y] Xb = X * beta;
    vector[N] Zg = Z * gamma;
    int ny = 1;
    for(n in 1:N) {
      if(D[n] > 0) {
        target += normal_lpdf(y[ny] | Xb[ny], sigma_e) + log(Phi((Zg[n] + rho / sigma_e * (y[ny] - Xb[ny])) / sqrt(1 - rho^2)));
        ny += 1; 
      }
      else {
        target += log(Phi(-Zg[n]));
      }
    }
  }
}
