// ex1.stan (Stan >= 2.24)
functions {
  // derivative function
  vector deriv(real time, vector y, real k) {
    vector[1] dydt = k * (1 - y);
    return dydt;
  }
}
data {
  // rate estimation
  int<lower = 1> n;
  vector[n] k_obs;  // observed rate constants (1/min)
  vector[n] temp_K; // observed temperatures (K)
  // ode integration
  int<lower = 1> n1;
  vector[1] y0;
  real times[n1];   // evaluated time points (min)
  real temp_K_new;  // evaluated temperature (K) 
}
parameters {
  real lnA;                 // pre-exponential factor
  real<lower = 0> Ea;       // activation energy
  real<lower = 1E-6> sigma; // error standard deviation                  
}
model {
  // (naive) priors
  lnA ~ normal(25, 10);
  Ea ~ normal(50, 10);
  sigma ~ normal(0, 1);
  // likelihood
  k_obs ~ normal(exp(lnA - Ea ./ (0.0083144 * temp_K)), sigma); 
}
generated quantities {
  vector[1] y_new[n1];
  // predict rate constant k-hat
  real k_hat = exp(lnA - Ea / (0.0083144 * temp_K_new));
  // integrate reaction model based on k-hat
  y_new = ode_rk45(deriv, y0, 0.0, times, k_hat); 
}
