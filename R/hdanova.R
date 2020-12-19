#' Hypothesis Test for High-dimensional Data
#' @description Test the mean or differences of means of high-dimensional vectors are zero or not.
#' @param X a matrix (one-sample) or a list of matrices (multiple-samples), with each row representing an observation.
#' @param alpha significance level; default value: 0.05.
#' @param side either of \code{'<=','>='} or \code{'=='}; default value: \code{'=='}.
#' @param tau real number(s) in the interval \code{[0,1)} that specifies the decay parameter and is automatically selected if it is set to \code{NULL} or multiple values are provided; default value: \code{NULL}, which is equivalent to \code{tau=1/(1+exp(-0.8*seq(-6,5,by=1))).}
#' @param B the number of bootstrap replicates; default value: \code{ceiling(50/alpha)}.
#' @param pairs a matrix with two columns, only used when there are more than two populations, where each row specifies a pair of populations for which the SCI is constructed; default value: \code{NULL}, so that SCIs for all pairs are constructed.
#' @param Sig a matrix (one-sample) or a list of matrices (multiple-samples), each of which is the covariance matrix of a sample; default value: \code{NULL}, so that it is automatically estimated from data.
#' @param verbose TRUE/FALSE, indicator of whether to output diagnostic information or report progress; default value: FALSE.
#' @param tau.method the method to select tau; possible values are 'MGB' (default), 'MGBA', 'RMGB', 'RMGBA', 'WB' and 'WBA' (see \code{\link{hdsci}}).
#' @param R the number of Monte Carlo replicates for estimating the empirical size; default: \code{ceiling(25/alpha)}
#' @param nblock the number of block in CUDA computation
#' @param tpb number of threads per block; the maximum number of total number of parallel GPU threads is then \code{nblock*tpb}
#' @param seed the seed for random number generator
#' @param sci T/F, indicating whether to construct SCIs or not; default: FALSE.
#' @return a list that includes all objects returned by \code{\link{hdsci}} and the following additional objects:
#'     \describe{
#'          \item{\code{tau}}{a vector of candidate values of the decay parameter.}
#'          \item{\code{side}}{the input \code{side}.}
#'          \item{\code{alpha}}{the input \code{alpha}.}
#'          \item{\code{pairs}}{a matrix of two columns, each row containing the a pair of indices of samples of which the SCI of the difference in mean is constructed.}
#'          \item{\code{sigma2}}{a vector (for one sample) or a list (for multiple samples) of vectors containing variance for each coordinate.}
#'          \item{\code{selected.tau}}{the selected value of the decay parameter from \code{tau}.}
#'          \item{\code{size.tau}}{the estimated size for each value in \code{tau}.}
#'          \item{\code{pvalue.tau}}{the p-value for each value in \code{tau}.}
#'          \item{\code{pvalue}}{the p-value of the test.}
#'          \item{\code{reject}}{a T/F value indicating whether the hypothesis is rejected.}
#'          \item{\code{rej.paris}}{optionally gives the pairs of samples that lead to rejection.}
#'          \item{\code{sci}}{if \code{return.sci=TRUE}, then a constructed SCI constructed by using \code{selected,tau}, which is a list of the following objects:
#'              \describe{
#'                  \item{\code{sci.lower}}{a vector (when <= two samples) or a list of vectors (when >= 3 samples) specifying the lower bound of the SCI for the mean (one-sample) or the difference of means of each pair of samples.}
#'                  \item{\code{sci.upper}}{a vector (when <= two samples) or a list of vectors (when >= 3 samples) specifying the upper bound of the SCI.}
#'                  \item{\code{pairs}}{a matrix of two columns, each row containing the a pair of indices of samples of which the SCI of the difference in mean is constructed.}
#'                  \item{\code{tau}}{the decay parameter that is used to construct the SCI.}
#'                  \item{\code{Mn}}{the sorted (in increasing order) bootstrapped max statistic.}
#'                  \item{\code{Ln}}{the sorted (in increasing order) bootstrapped min statistic.}
#'                  \item{\code{side}}{the input \code{side}.}
#'                  \item{\code{alpha}}{the input \code{alpha}.}
#'          }
#' @importFrom Rdpack reprompt
#' @import Rcpp
#' @useDynLib hdanova.cuda
#' @references 
#' \insertRef{Lopes2020}{hdanova}
#' 
#' \insertRef{Lin2020}{hdanova}
#' @examples
#' # simulate a dataset of 1 sample
#' hdtest(X=matrix(runif(30*100)-0.5,30,100))
#' @export
hdtest <- function(X,alpha=0.05,side='==',tau=NULL,B=ceiling(50/alpha),pairs=NULL,Sig=NULL,verbose=F,tau.method='MGB',
                   R=ceiling(25/alpha),nblock=32,tpb=64,seed=sample.int(2^30,1),return.sci=F)
{
    sci.side <- switch (side,
                        '>=' = 'upper',
                        '<=' = 'lower',
                        '==' = 'both',
                        '='  = 'both',
                        'both' = 'both'
    )
    
    S <- hdanova(X, alpha, sci.side, tau, B, pairs, Sig, verbose, tau.method, R, nblock, tpb, seed)
    
    
    if(is.null(S)) return(NULL)
    
    res <- list(tau=S$tau,
                alpha=alpha,
                side=side,
                sigma2=lapply(1:nrow(S$sigma), function(k) S$sigma[k,]^2),
                selected.tau=S$selected.tau,
                pvalue.tau=S$pvalue.tau,
                size.tau=S$size.tau,
                pairs=S$pairs)
    

    if(return.sci)
        res$sci <- hdsciK(X, alpha, sci.side, S$tau,B, S$pairs, verbose, S$Mn, S$Ln, S$sigma^2, S$selected.tau)$sci

    
    res$pvalue <- res$pvalue.tau[which(res$tau==res$selected.tau)]
    
    res$reject <- (res$pvalue < alpha)

    
    if(S$K > 2)
    {
        if('sci' %in% names(res)) sci <- res$sci
        else sci <- hdsciK(X, alpha, sci.side, S$tau,B, S$pairs, verbose, S$Mn, S$Ln, S$sigma^2, S$selected.tau)$sci
        rej.idx <- sapply(sci$sci.lower,function(z) any(z>0)) |
            sapply(sci$sci.upper, function(z) any(z<0))
        if(res$reject)
        {
            rej.pairs <- sci$pairs[rej.idx,]
            res$rej.pairs <- rej.pairs
        }
        
        tmp <- cbind(sci$pairs,rej.idx)
        colnames(tmp) <- c('g1','g2','reject')
        res$pairs <- tmp
    }
    
    class(res) <- 'hdaov'
    attr(res,'vnames') <- attr(S,'vnames')
    return(res)
}



#' Empirical Size Associated with Decay Parameter
#' @description Find the empirical size associated with the decay parameter conditional on a dataset
#' @param X a matrix (one-sample) or a list of matrices (multiple-samples), with each row representing an observation.
#' @param alpha significance level; default value: 0.05.
#' @param side either of \code{'<=','>='} or \code{'=='}; default value: \code{'=='}.
#' @param tau real number(s) in the interval \code{[0,1)} for which the empirical size will be evaluated.
#' @param B the number of bootstrap replicates; default value: \code{ceiling(50/alpha)}.
#' @param pairs a matrix with two columns, only used when there are more than two populations, where each row specifies a pair of populations for which the SCI is constructed; default value: \code{NULL}, so that SCIs for all pairs are constructed.
#' @param verbose TRUE/FALSE, indicator of whether to output diagnostic information or report progress; default value: FALSE.
#' @param method the evaluation method tau; possible values are 'MGB' (default), 'MGBA', 'RGB', 'RGBA', 'WB' and 'WBA' (see \code{\link{hdsci}} for details).
#' @param R the number of iterations; default value: \code{ceiling(25/alpha)}.
#' @param nblock the number of block in CUDA computation
#' @param tpb number of threads per block; the maximum number of total number of parallel GPU threads is then \code{nblock*tpb}
#' @param seed the seed for random number generator
#' @return a vector of empirical size corresponding to \code{tau}.
#' @importFrom Rdpack reprompt
#' @references 
#' \insertRef{Lopes2020}{hdanova}
#' 
#' \insertRef{Lin2020}{hdanova}
#' @examples
#' # simulate a dataset of 4 samples
#' X <- lapply(1:4, function(g) MASS::mvrnorm(30,rep(0.3*g,10),diag((1:10)^(-0.5*g))))
#' 
#' size.tau(X,tau=seq(0,1,by=0.1),alpha=0.05,pairs=matrix(1:4,2,2),R=100)
#' @export
size.tau <- function(X,tau,side='==',alpha=0.05,B=ceiling(50/alpha),pairs=NULL,verbose=F,
                     method='MGB',R=ceiling(25/alpha),
                     nblock=32,tpb=64,seed=sample.int(2^30,1))
{
    side <- switch (side,
                        '>=' = 'upper',
                        '<=' = 'lower',
                        '==' = 'both',
                        '='  = 'both',
                        'both' = 'both'
    )
    res <- hdanova(X, alpha, side, tau, B, pairs, Sig, verbose, method, R, nblock, tpb, seed)
    return(res$size.tau)
}
    