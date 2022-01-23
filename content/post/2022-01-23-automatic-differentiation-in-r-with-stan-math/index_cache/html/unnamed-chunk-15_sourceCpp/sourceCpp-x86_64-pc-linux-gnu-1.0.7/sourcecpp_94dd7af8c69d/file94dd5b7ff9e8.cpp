// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::plugins(cpp14)]]

using namespace Rcpp;

// [[Rcpp::export]]
auto fjac_exp(Eigen::VectorXd x, Eigen::VectorXd theta)
{
  // declarations
  Eigen::VectorXd fx;
  Eigen::MatrixXd jac_fx;
  
  // response and jacobian
  stan::math::jacobian([&x](auto theta) { 
    // exponential model 
    return stan::math::add(theta(0) * stan::math::exp(-theta(1) * x), theta(2)); 
  }, theta, fx, jac_fx);
  
  // reformat returned result
  NumericVector fx1 = wrap(fx);
  NumericMatrix jac_fx1 = wrap(jac_fx);
  colnames(jac_fx1) = CharacterVector({"A", "lam", "b"});
  fx1.attr("gradient") = jac_fx1;
  
  return fx1;
}


#include <Rcpp.h>
#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// fjac_exp
auto fjac_exp(Eigen::VectorXd x, Eigen::VectorXd theta);
RcppExport SEXP sourceCpp_9_fjac_exp(SEXP xSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(fjac_exp(x, theta));
    return rcpp_result_gen;
END_RCPP
}
