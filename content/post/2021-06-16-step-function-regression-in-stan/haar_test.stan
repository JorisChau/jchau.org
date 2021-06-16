functions{
  #include haar.stan 
}
data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
}
generated quantities {
   vector[N] ywd = fwt(y);
   vector[N] y1 = iwt(ywd);
}
