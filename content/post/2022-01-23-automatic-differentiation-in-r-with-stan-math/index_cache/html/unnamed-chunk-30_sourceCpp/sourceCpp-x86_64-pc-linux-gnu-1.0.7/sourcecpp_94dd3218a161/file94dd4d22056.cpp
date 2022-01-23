// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::plugins(cpp14)]]

using namespace Rcpp;
using stan::math::exp;
using stan::math::fmax;
using stan::math::fmin;
using stan::math::log1m;
using stan::math::multiply_log;

struct kinetic_func {
    template <typename T_t, typename T_y, typename T_theta>
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_theta>, Eigen::Dynamic, 1>
    operator()(const T_t &t, const Eigen::Matrix<T_y, Eigen::Dynamic, 1> &y,
               std::ostream *msgs, const Eigen::Matrix<T_theta, Eigen::Dynamic, 1> &theta) const {

        Eigen::Matrix<T_y, Eigen::Dynamic, 1> dydt(1);

        T_y y1 = fmin(fmax(y(0), 1e-10), 1.0 - 1e-10); // constrain y to unit interval
        dydt << exp(theta(0) + theta(1) * log1m(y1) + multiply_log(theta(2), y1) + multiply_log(theta(3), -log1m(y1))); 
    
        return dydt;
    }
};

// [[Rcpp::export]]
auto fjac_kinetic(double logk, double n, double m, double p, NumericVector ts)
{
    // initialization
    Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> theta(4);
    theta << logk, n, m, p;

    Eigen::VectorXd y0(1);
    y0 << 0.001;

    kinetic_func kf;

    // ode integration
    auto ys = stan::math::ode_rk45(kf, y0, 0, as<std::vector<double>>(ts), 0, theta);

    Eigen::VectorXd fx(ts.length());
    Eigen::MatrixXd jac_fx(4, ts.length());

    // response and jacobian
    for (int n = 0; n < ts.length(); ++n) {
        stan::math::set_zero_all_adjoints();
        ys[n](0).grad();
        fx(n) = ys[n](0).val();
        jac_fx.col(n) = theta.adj();
    }

    // reformat returned result
    NumericVector fx1 = wrap(fx);
    jac_fx.transposeInPlace();
    NumericMatrix jac_fx1 = wrap(jac_fx);
    colnames(jac_fx1) = CharacterVector({"logk", "n", "m", "p"});
    fx1.attr("gradient") = jac_fx1;

    return fx1;
}


#include <Rcpp.h>
#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// fjac_kinetic
auto fjac_kinetic(double logk, double n, double m, double p, NumericVector ts);
RcppExport SEXP sourceCpp_5_fjac_kinetic(SEXP logkSEXP, SEXP nSEXP, SEXP mSEXP, SEXP pSEXP, SEXP tsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type logk(logkSEXP);
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ts(tsSEXP);
    rcpp_result_gen = Rcpp::wrap(fjac_kinetic(logk, n, m, p, ts));
    return rcpp_result_gen;
END_RCPP
}
