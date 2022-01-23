// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::plugins(cpp14)]]

using namespace Rcpp;
using stan::math::add;
using stan::math::multiply;
using stan::math::square;
using stan::math::subtract;

struct watson_func {

    // members
    const size_t n_;
    Eigen::MatrixXd tj1, tj2;

    // constructor
    watson_func(size_t n = 31, size_t p = 6) : n_(n) {

      tj1.resize(n - 2, p);
      tj2.resize(n - 2, p);

      double tj, ti;
      for (int i = 0; i < n - 2; ++i) {
        ti = (i + 1) / 29.0;
        tj = 1.0;
        tj1(i, 0) = tj;
        tj2(i, 0) = 0.0;
        for (int j = 1; j < p; ++j) {
          tj2(i, j) = j * tj;
          tj *= ti;
          tj1(i, j) = tj;
        }
      }

    }

    // function definition
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &theta) const {

      Eigen::Matrix<T, Eigen::Dynamic, 1> fx(n_);

      fx << subtract(multiply(tj2, theta), add(square(multiply(tj1, theta)), 1.0)),
          theta(0), theta(1) - theta(0) * theta(0) - 1.0;

      return fx;
    }
};

// [[Rcpp::export]]
auto fjac_watson(Eigen::VectorXd theta, CharacterVector nms) {

  // declarations
  Eigen::VectorXd fx;
  Eigen::MatrixXd jac_fx;
  watson_func wf;

  // response and jacobian
  stan::math::jacobian(wf, theta, fx, jac_fx);

  // reformat returned result
  NumericVector fx1 = wrap(fx);
  NumericMatrix jac_fx1 = wrap(jac_fx);
  colnames(jac_fx1) = nms;
  fx1.attr("gradient") = jac_fx1;

  return fx1;
}
