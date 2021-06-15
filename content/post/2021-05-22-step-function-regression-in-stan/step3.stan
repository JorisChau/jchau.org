// naive step function regression
data {
  int<lower=1> N;
  vector[N] ywd;
  real tau;
  real<lower = 0> sigma0;
}
transformed data{
  // real tau = 1E-6;
  real c = 10 / tau;
}
parameters {
  vector[N] d;
  vector<lower = 0, upper = 1>[N] p;
  real<lower=0> sigma;
}

model {

  p ~ beta(0.25, 0.25);
  sigma ~ normal(sigma0, 0.01);

  for(n in 1:N)
    target += log_mix(p[n], normal_lpdf(d[n] | 0, c * tau), normal_lpdf(d[n] | 0, tau));

  ywd ~ normal(d, sigma);
}
