#' Construct Simultaneous Confidence Interval
#' @description Construct (1-\code{alpha}) simultaneous confidence interval (SCI)  for the mean or difference of means of high-dimensional vectors.
#' @param X a matrix (one-sample) or a list of matrices (multiple-samples), with each row representing an observation.
#' @param alpha significance level; default value: 0.05.
#' @param side either of \code{'lower','upper'}, or \code{'both'}; default value: \code{'both'}.
#' @param tau real number(s) in the interval \code{[0,1)} that specifies the decay parameter and is automatically selected if it is set to \code{NULL} or multiple values are provided; default value: \code{NULL}, which is equivalent to \code{tau=1/(1+exp(-0.8*seq(-6,5,by=1))).}
#' @param B the number of bootstrap replicates; default value: \code{ceiling(50/alpha)}.
#' @param pairs a matrix with two columns, only used when there are more than two populations, where each row specifies a pair of populations for which the SCI is constructed; default value: \code{NULL}, so that SCIs for all pairs are constructed.
#' @param Sig a matrix (one-sample) or a list of matrices (multiple-samples), each of which is the covariance matrix of a sample; default value: \code{NULL}, so that it is automatically estimated from data.
#' @param verbose TRUE/FALSE, indicator of whether to output diagnostic information or report progress; default value: FALSE.
#' @param tau.method the method to select tau; possible values are 'MGB' (default), 'MGBA', 'RMGB', 'RMGBA', 'WB' and 'WBA' (see details).
#' @param R the number of Monte Carlo replicates for estimating the empirical size; default: \code{ceiling(25/alpha)}
#' @param nblock the number of block in CUDA computation
#' @param tpb number of threads per block; the maximum number of total number of parallel GPU threads is then \code{nblock*tpb}
#' @param seed the seed for random number generator
#' @return a list of the following objects: 
#'      \describe{
#'          \item{\code{sci}}{the constructed SCI, which is a list of the following objects:
#'              \describe{
#'                  \item{\code{sci.lower}}{a vector (when <= two samples) or a list of vectors (when >= 3 samples) specifying the lower bound of the SCI for the mean (one-sample) or the difference of means of each pair of samples.}
#'                  \item{\code{sci.upper}}{a vector (when <= two samples) or a list of vectors (when >= 3 samples) specifying the upper bound of the SCI.}
#'                  \item{\code{pairs}}{a matrix of two columns, each row containing the a pair of indices of samples of which the SCI of the difference in mean is constructed.}
#'                  \item{\code{tau}}{the decay parameter that is used to construct the SCI.}
#'                  \item{\code{Mn}}{the sorted (in increasing order) bootstrapped max statistic.}
#'                  \item{\code{Ln}}{the sorted (in increasing order) bootstrapped min statistic.}
#'                  \item{\code{side}}{the input \code{side}.}
#'                  \item{\code{alpha}}{the input \code{alpha}.}
#'              }
#'          }
#'          \item{\code{tau}}{a vector of candidate values of the decay parameter.}
#'          \item{\code{sci.tau}}{a list of \code{sci} objects corresponding to the candidate values in \code{tau}.}
#'          \item{\code{selected.tau}}{the selected value of the decay parameter from \code{tau}.}
#'          \item{\code{side}}{the input \code{side}.}
#'          \item{\code{alpha}}{the input \code{alpha}.}
#'          \item{\code{pairs}}{a matrix of two columns, each row containing the a pair of indices of samples of which the SCI of the difference in mean is constructed.}
#'          \item{\code{sigma2}}{a vector (for one sample) or a list (for multiple samples) of vectors containing variance for each coordinate.}
#'          }
#' @details Four methods to select the decay parameter \code{tau} are provided. Using the fact that a SCI is equivalent to a hypothesis test problem, all of them first identify a set of good candidates which give rise to test that respects the specified level \code{alpha}, and then select a candidate that minimizes the p-value. These methods differ in how to identify the good candidates.
#'     \describe{
#'         \item{\code{MGB}}{for this method, conditional on the data \code{X}, \code{B0=10*ceiling(1/alpha)} i.i.d. zero-mean multivariate Gaussian samples (called MGB samples here) are drawn, where the covariance of each sample is equal to the sample covariance matrix \code{Sig} of the data \code{X}. For each candidate value in \code{tau}, 1) the empirical distribution of the corresponding max/min statistic is obtained by reusing the same bootstrapped sample, 2) the corresponding p-value is obtained, and 3) the size is estimated by applying the test to all MGB samples. The candidate values with the empirical size closest to \code{alpha} are considered as good candidates.}
#'         \item{\code{MGBA}}{an slightly more aggressive version of \code{MGB}, where the candidate values with the estimated empirical size no larger than \code{alpha} are considered good candidates.}    
#'         \item{\code{RMGB}}{this method is similar to \code{MGB}, except that for each MGB sample, the covariance matrix is the sample covariance matrix of a resampled (with replacement) data \code{X}.}
#'         \item{\code{RMGBA}}{an slightly more aggressive version of \code{RMGB}, where the candidate values with the estimated empirical size no larger than \code{alpha} are considered good candidates.}
#'         \item{\code{WB}}{for this method, conditional on \code{X}, \code{B0=10*ceiling(1/alpha)} i.i.d. samples (called WB samples here) are drawn by resampling \code{X} with replacement. For each candidate value in \code{tau}, 1) the corresponding p-value is obtained, and 2) the size is estimated by applying the test to all WB samples without reusing the bootstrapped sample. The candidate values with the empirical size closest to \code{alpha} are considered as good candidates.}
#'         \item{\code{WBA}}{an slightly more aggressive version of \code{WB}, where the candidate values with the estimated empirical size no larger than \code{alpha} are considered good candidates.}
#'     }
#'     Among these methods, MGB and MGBA are recommended, since they are computationally more efficiently and often yield good performance. The MGBA might have slightly larger empirical size. The WB and WBA methods may be subject to outliers, in which case they become more conservative. The RMGB is computationally slightly slower than WB, but is less subject to outliers.
#' @importFrom Rdpack reprompt
#' @references 
#' \insertRef{Lopes2020}{hdanova}
#' 
#' \insertRef{Lin2020}{hdanova}
#' @examples  
#' # simulate a dataset of 4 samples
#' X <- lapply(1:4, function(g) MASS::mvrnorm(30,rep(0,10),diag((1:10)^(-0.5*g))))
#' 
#' # construct SCIs for the mean vectors with pairs={(1,3),(2,4)}
#' hdsci(X,alpha=0.05,pairs=matrix(1:4,2,2))$sci
#' @export
hdsci <- function(X,alpha=0.05,side='both',tau=NULL,B=ceiling(50/alpha),pairs=NULL,
                  Sig=NULL,verbose=F,tau.method='MGB',R=ceiling(25/alpha),
                  nblock=32,tpb=64,seed=sample.int(2^30,1))
{
    S <- hdanova(X, alpha, side, tau, B, pairs, Sig, verbose, tau.method, R, nblock, tpb, seed)
    
    if(is.null(S)) return(NULL)
    
    #if(S$K==1) res <- hdsci1(X, alpha, side, S$tau, B, verbose, S$Mn, S$Ln, S$sigma^2, S$selected.tau)
    #else 
        res <- hdsciK(X, alpha, side, S$tau, B, S$pairs, verbose, S$Mn, S$Ln, S$sigma^2, S$selected.tau)
    
    return(res)
}

hdanova <- function(X,alpha=0.05,side='both',tau=NULL,B=ceiling(50/alpha),pairs=NULL,
                    Sig=NULL,verbose=F,tau.method='MGB',R=ceiling(25/alpha),
                    nblock=32,tpb=64,seed=sample.int(2^30,1))
{
    if(is.null(tau)) tau=1/(1+exp(-0.8*seq(-6,5,by=1)))
    
    if(tau.method %in% c('MGB','MGBA')){
        method <- 0
        mod <- ifelse(tau.method=='MGB',yes='C',no='A')
    }     else if(tau.method %in% c('RMGB','RMGBA')){
        method <- 1
        mod <- ifelse(tau.method=='RMGB',yes='C',no='A')
    }     else{
        method <- 2
        mod <- ifelse(tau.method=='WB',yes='C',no='A')
    }
    
    R <- ifelse(length(tau)==1,yes=0,no=R)
    
    if(is.matrix(X)) # one-sample
    {
        K <- 1
        ns <- nrow(X)
        p <- ncol(X)
        sigma <- matrix( apply(X,2,sd),1,p)
        mu <-  matrix(apply(X,2,mean),1,p)
        cudaX <- X
        pairs <- NULL
        
    }
    else if(is.list(X)) # now 2 or more samples
    {
        K <- length(X)
        
        ns <- sapply(X,function(x){nrow(x)}) 
        p <- ncol(X[[1]]) 
        n <- sum(ns) 
        mu <- lapply(X,function(x) apply(x,2,mean))
        mu <- do.call("rbind", mu)
        
        sigma <- lapply(X, function(x) apply(x, 2, sd))
        sigma <- do.call("rbind",sigma)
        
        cudaX <- do.call("rbind",X)
        
        if(is.null(pairs)) pairs <- t(combn(1:K,2))
        
    }
    else stop('X must be matrix or a list')
    
    side <- switch(side,
                       'both'=0,
                       'lower'=-1,
                       'upper'=1)
    
    S <- anova_cuda(cudaX,c(ns),sigma,mu,tau,pairs,side,B,alpha,method,R,nblock,tpb,seed=seed) 
    
    if(is.null(S)){
        message('Some error occurs in GPU computation. NULL will be returned')
        return(NULL)
    } 
    
    if(length(tau) == 1) S$selected.tau <- tau
    else S$selected.tau <- choose.tau(alpha, tau, S$size.tau, S$pvalue.tau, mod=mod, margin=0.01)
    
    
    S$sigma <- sigma
    S$mu <- mu
    S$K <- K
    S$tau <- tau
    S$pairs <- pairs
    
    return(S)
}

# for one sample
# if tau.method==NULL, then no selection of tau is made
# hdsci1 <- function(X,alpha,side,tau,B,verbose,Mn,Ln,sigma2,selected.tau)
# {
#     n <- nrow(X)
#     p <- ncol(X)
#     
#     rtn <- sqrt(n)
#     
#     X.bar <- apply(X,2,mean)
#     if(is.null(sigma2)) sigma2 <- apply(X,2,var)
#     
#     sci.tau <- lapply(1:length(tau),function(v){
#         
#         # construct SCI
#         side <- tolower(side)
#         if(side == 'both')
#         {
#             a1 <- alpha/2
#             a2 <- 1 - alpha/2
#         }
#         else
#         {
#             a1 <- alpha
#             a2 <- alpha
#         }
#         
#         b1 <- max(1,round(a1*B))
#         b2 <- round(a2*B)
#         
#         sigma <- sqrt(sigma2)^tau[v]
#         
#         idx <- sigma == 0
#         
#         sci.lower <- X.bar - Mn[v,b2] * sigma / rtn
#         sci.lower[idx] <- 0
#         sci.upper <- X.bar - Ln[v,b1] * sigma / rtn
#         sci.upper[idx] <- 0
#         
#         if(side == 'upper')  sci.lower[] <- -Inf
#         if(side == 'lower')  sci.upper[] <- Inf
#         
#         list(sci.lower=sci.lower,
#              sci.upper=sci.upper,
#              sigma2=sigma2,
#              tau=tau[v],
#              side=side,
#              alpha=alpha,
#              Mn=Mn[v,],
#              Ln=Ln[v,])
#     })
#     
#     v <- which(tau==selected.tau)
#     
#     # output
#     res <- list(tau=tau,
#                 alpha=alpha,
#                 side=side,
#                 sigma2=sigma2,
#                 pairs=NULL,
#                 sci.tau=sci.tau,
#                 sci=sci.tau[[v]],
#                 selected.tau=selected.tau)
# 
#     return(res)
# }

# for more than one sample
# if tau.method==NULL, then no selection of tau is made
hdsciK <- function(X,alpha,side,tau,B,pairs,verbose,Mn,Ln,sigma2,selected.tau)
{
    
    if(is.matrix(X)) # one-sample
    {
        n <- nrow(X)
        p <- ncol(X)
        
        rtn <- sqrt(n)
        
        X.bar <- apply(X,2,mean)
        if(is.null(sigma2)) sigma2 <- apply(X,2,var)
        
        sci.tau <- lapply(1:length(tau),function(v){
            
            # construct SCI
            side <- tolower(side)
            if(side == 'both')
            {
                a1 <- alpha/2
                a2 <- 1 - alpha/2
            }
            else
            {
                a1 <- alpha
                a2 <- alpha
            }
            
            b1 <- max(1,round(a1*B))
            b2 <- round(a2*B)
            
            sigma <- sqrt(sigma2)^tau[v]
            
            idx <- sigma == 0
            
            sci.lower <- X.bar - Mn[v,b2] * sigma / rtn
            sci.lower[idx] <- 0
            sci.upper <- X.bar - Ln[v,b1] * sigma / rtn
            sci.upper[idx] <- 0
            
            if(side == 'upper')  sci.lower[] <- -Inf
            if(side == 'lower')  sci.upper[] <- Inf
            
            list(sci.lower=sci.lower,
                 sci.upper=sci.upper,
                 sigma2=sigma2,
                 tau=tau[v],
                 side=side,
                 alpha=alpha,
                 Mn=Mn[v,],
                 Ln=Ln[v,])
        })
    }
    else
    {
        ns <- sapply(X,function(x){nrow(x)}) # size of each sample
        p <- ncol(X[[1]]) # dimension
        n <- sum(ns) # total sample size
        G <- length(X) # number of samples
        
        sci.tau <- lapply(1:length(tau),function(v){
            # construct SCI
            side <- tolower(side)
            if(side == 'both')
            {
                a1 <- alpha/2
                a2 <- 1 - alpha/2
            }
            else
            {
                a1 <- alpha
                a2 <- alpha
            }
            
            
            b1 <- max(1,round(a1*B))
            b2 <- round(a2*B)
            
            sci.lower <- list()
            sci.upper <- list()
            
            for(q in 1:nrow(pairs))
            {
                j <- pairs[q,1]
                k <- pairs[q,2]
                
                lamj <- sqrt(ns[k]/(ns[j]+ns[k]))
                lamk <- sqrt(ns[j]/(ns[j]+ns[k]))
                
                sig2j <- sigma2[j,]
                sig2k <- sigma2[j,]
                
                sigjk <- sqrt(lamj^2 * sig2j + lamk^2 * sig2k)^tau[v] 
                
                X.bar <- apply(X[[j]],2,mean)
                Y.bar <- apply(X[[k]],2,mean)
                sqrt.harm.n <- sqrt(ns[j]*ns[k]/(ns[j]+ns[k]))
                
                idx <- (sigjk==0)
                
                if(side == 'both' || side == 'lower')
                {
                    tmp <- (X.bar-Y.bar) - Mn[v,b2] * sigjk / sqrt.harm.n
                    tmp[idx] <- 0
                    sci.lower[[q]] <- tmp
                }
                if(side == 'both' || side == 'upper')
                {
                    tmp <- (X.bar-Y.bar) - Ln[v,b1] * sigjk / sqrt.harm.n
                    tmp[idx] <- 0
                    sci.upper[[q]] <- tmp
                }
                
                if(side == 'upper')  sci.lower[[q]] <- rep(-Inf,p)
                if(side == 'lower')  sci.upper[[q]] <- rep(Inf,p)
            }
            
            if(G <= 2) 
            {
                sci.lower <- sci.lower[[1]]
                sci.upper <- sci.upper[[1]]
            }
            
            list(sci.lower=sci.lower,
                 sci.upper=sci.upper,
                 sigma2=sigma2,
                 tau=tau[v],
                 side=side,
                 alpha=alpha,
                 pairs=pairs,
                 Mn=Mn[v,],
                 Ln=Ln[v,])
        })
    }
    
    
    v <- which(tau==selected.tau)
    
    # output
    res <- list(tau=tau,
                alpha=alpha,
                side=side,
                sigma2=sigma2,
                pairs=pairs,
                sci.tau=sci.tau,
                sci=sci.tau[[v]],
                selected.tau=selected.tau)
    
    return(res)
}

# choose tau based on the estimated size and calculuated p-values
choose.tau <- function(alpha,tau,size,pv,mod='C',margin=0.01)
{
    if(mod == 'C')
    {
        s <- abs(size-alpha)
        smin <- min(s)
        
        # those with empirical size closest to alpha are "good" candidate tau values
        # allow a margin to account for computer finite-precision
        idx <- which( abs(s-smin) < margin*alpha) 
    }
    else # aggressive version, considering all candidate values with empirical size lower than alpha whenever possible
    {
        # no empirical sizes lower than alpha
        if(all(size > (1+margin)*alpha) == 0)
        {
            idx <- which(size==min(size))
        }
        else # those with empirical size lower than alpha are "good" candidate values
        {
            idx <- which(size <= (1+margin)*alpha)
        }
    }
    
    # consider only good candidate values
    pv[-idx] <- Inf 
    
    # now select tau to minimize p-value
    # when there are multiple such tau values, take the mean of them
    tau0 <- mean(tau[which(pv==min(pv))]) 
    
    # If tau0 is the mean of two or more candidate values,
    # then tau0 might not be in the prescribed list
    # In this case, we select a tau in the list that is closest to tau0
    tau[which.min(abs(tau-tau0))] 
}

