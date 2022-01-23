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
