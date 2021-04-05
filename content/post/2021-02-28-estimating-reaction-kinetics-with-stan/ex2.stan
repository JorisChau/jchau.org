// ex2.stan (Stan >= 2.24)
functions {
  vector deriv(real time, vector y, vector ks, real m) {
    int S = rows(ks);
    vector[S] dydt = ks .* pow(1 - y, rep_vector(m, S));
    return dydt;
  }
}
data {
  int<lower = 1> n;                       // # observations
  int<lower = 1> S;                       // # replicates
  real times[n];                          // observed times
  vector<lower = 0, upper = 1>[S] y_s[n]; // observed trends
}
transformed data {
  vector[S] y0 = rep_vector(0.0, S);
}
parameters {
  real<lower = 0> k;                 // mean rate constant
  real<lower = 0, upper = 3> m;      // reaction order
  real<lower = 0> sigma_u;           // replicate sd
  real<lower = 1E-10> phi_inv;       // residual sd scale
  vector[S] u_s;                     // random effects          
}
transformed parameters{
  vector[S] ks;
  vector[S] alpha_s[n];
  ks = k + sigma_u * u_s;
  alpha_s = ode_rk45(deriv, y0, 0, times, ks, m); 
}
model {
  // (naive) normal priors
  k ~ normal(0, 1);
  m ~ normal(1, 1);
  sigma_u ~ normal(0, 1);
  phi_inv ~ normal(0, 1);
  // likelihood
  u_s ~ std_normal();
  for(s in 1:S) 
    y_s[][s] ~ beta(alpha_s[][s] / phi_inv, (1 - alpha_s[][s]) / phi_inv);
}
