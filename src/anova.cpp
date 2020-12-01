#include <Rcpp.h>
using namespace Rcpp;

extern "C"
int anova_cuda_(const NumericMatrix& vX, 
              const NumericVector& vN, 
              const NumericMatrix& sigma, 
              const NumericMatrix& mu,
              const NumericVector& vtau, 
              Nullable<NumericMatrix> vpair,
              int side,
              int B, 
              double alpha,
              int method,
              int R,
              int nblock, 
              int threads_per_block,
              int seed,
              NumericVector& pval,
              NumericVector& size,
              NumericMatrix& vMn,
              NumericMatrix& vLn);

//[[Rcpp::export]]
SEXP anova_cuda(const NumericMatrix& vX, 
                 const NumericVector& vN, 
                 const NumericMatrix& vsigma, 
                 const NumericMatrix& mu,
                 const NumericVector& vtau, 
                 Nullable<NumericMatrix> vpair,
                 int side,
                 int B, 
                 double alpha,
                 int method,
                 int R,
                 int nblock, 
                 int threads_per_block,
                 int seed) {
    //S4 c(r);
    //double *x = REAL(c.slot("x"));
    //int *y = INTEGER(c.slot("y"));
    //x[0] = 500.0;
    //y[1] = 1000;
    
    NumericVector pval(vtau.size(),-1);
    NumericVector size(vtau.size(),-1);
    NumericMatrix Mn(vtau.size(),B);
    NumericMatrix Ln(vtau.size(),B);
    
    int code = anova_cuda_(vX,vN,vsigma,mu,vtau,vpair,side,B,alpha,method,R,nblock,threads_per_block,seed,pval,size,Mn,Ln);
    if(code > 0) return R_NilValue;
    else
        return Rcpp::List::create(Rcpp::Named("pvalue.tau") = pval,
                              Rcpp::Named("size.tau") = size,
                              Rcpp::Named("Mn") = Mn,
                              Rcpp::Named("Ln") = Ln);
}
