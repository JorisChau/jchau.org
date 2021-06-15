data {
  int<lower=1> N;
  int<lower=1> K;
  vector[N] x;
  vector[N] y;
}
transformed data {
  simplex[K + 1] gamma_inc = rep_vector(0.25, 4);
}
parameters {
  real mu[K + 1];
  real<lower = 0> sigma;
  // simplex[K + 1] gamma_inc;
}
transformed parameters {
  vector[K + 2] gamma = append_row(0, cumulative_sum(gamma_inc));
  vector[N] f;
  for(n in 1:N) {
    for(k in 1:(K + 1)) {
      if(x[n] > gamma[k] && x[n] <= gamma[k + 1]) {
        f[n] = mu[k];
      }
    }
  }
}
model {
  mu ~ normal(0, 5);
  sigma ~ exponential(1);
  // gamma_inc ~ dirichlet(rep_vector(1, K + 1));
  y ~ normal(f, sigma);
}

