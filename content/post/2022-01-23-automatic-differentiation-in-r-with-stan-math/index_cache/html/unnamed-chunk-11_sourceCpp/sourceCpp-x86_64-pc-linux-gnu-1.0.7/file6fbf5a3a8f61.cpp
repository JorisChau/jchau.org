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
auto grad_poly(double x, Eigen::VectorXd theta)
{
  // declarations
  double fx;
  Eigen::VectorXd grad_fx;
  // gradient calculation
  stan::math::gradient([&x](auto theta) {
    // polynomial function
    auto y = theta[0];
    for(int k = 1; k < 10; k++) {
      y += theta[k] * std::pow(x, k);
    }
    return y;
  }, theta, fx, grad_fx);
  // return gradient
  return grad_fx;
}
