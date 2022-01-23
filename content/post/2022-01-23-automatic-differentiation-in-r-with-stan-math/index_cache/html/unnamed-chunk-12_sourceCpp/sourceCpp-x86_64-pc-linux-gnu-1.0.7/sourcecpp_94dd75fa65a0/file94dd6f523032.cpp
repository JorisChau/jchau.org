// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>  // pulls in everything from rev/ and prim/
#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::plugins(cpp14)]]

// [[Rcpp::export]]
auto grad_poly(double x, Eigen::VectorXd theta)
{
  // declarations
  double fx;
  Eigen::VectorXd grad_fx;
  
  // gradient calculation
  stan::math::gradient([&x](auto theta) {
    // polynomial function
    auto y = theta[0];
    for(int k = 1; k < theta.size(); k++) {
      y += theta[k] * std::pow(x, k);
    }
    return y;
  }, theta, fx, grad_fx);
  
  // evaluated gradient
  return grad_fx;
}


#include <Rcpp.h>
#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// grad_poly
auto grad_poly(double x, Eigen::VectorXd theta);
RcppExport SEXP sourceCpp_5_grad_poly(SEXP xSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_poly(x, theta));
    return rcpp_result_gen;
END_RCPP
}
