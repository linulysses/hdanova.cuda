#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <Rcpp.h>
#include <limits>
#include <math.h>  // for using pow()
using namespace Rcpp;

#include <stdlib.h>     /* qsort */

// Macros to mimic 2D array by using 1D array
#define SIGMA(g,j) sigma[g*p+j]
#define MN(k,b) Mn[k*B+b]
#define LN(k,b) Ln[k*B+b]
#define MU(r,j) mu[r*p+j]
#define PAIRS(s,q) pairs[2*s+q]

// macros for negative and positive infinity
#define NEGATIVE_INF __longlong_as_double(0xfff0000000000000)
#define POSITIVE_INF __longlong_as_double(0x7ff0000000000000ULL)


#define GPU_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline bool gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
        return false;
    }
    else return true;
}


__global__ 
void setup_kernel(curandStateMRG32k3a* state, unsigned long long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
    number, no offset */
    curand_init(seed, id, 0, &state[id]);
}


__device__
void bootstrap_dev(double* X, int G, int p, int* N, int* pairs, double* sigma,
    double* tau, int m, int B, int b, curandStateMRG32k3a* localState,
    double* Mn, double* Ln, int *Xstar, double* z) {


        // initialize
        for (int k = 0; k < m; k++)
        {
            LN(k, b) = POSITIVE_INF; //__longlong_as_double(0x7ff0000000000000ULL); // Inf
            MN(k, b) = NEGATIVE_INF; // __longlong_as_double(0xfff0000000000000); // -Inf
        }

        //printf("kernel called: b=%d, N[0]=%f\n",b,N[0]);

        if (G == 1) // one-sample
        {
            for (int i = 0; i < N[0]; i++)
            {
                z[i] = curand_normal_double(localState);
            }

            double u, tmp, sigtau;
            for (int j = 0; j < p; j++)
            {
                u = 0;
                for (int i = 0; i < N[0]; i++)
                {
                    if(Xstar == nullptr)
                        u += z[i] * X[i*p + j];
                    else
                        u += z[i] * X[Xstar[i]*p + j];
                }


                for (int k = 0; k < m; k++)
                {
                    if (sigma[j] > 0)
                    {
                        sigtau = pow(sigma[j], tau[k]);
                        tmp = u / sigtau;

                        if (tmp > MN(k, b)) MN(k, b) = tmp;
                        if (tmp < LN(k, b)) LN(k, b) = tmp;
                    }

                }

            }

            double sqrt_n = sqrt((double)N[0]);
            for (int k = 0; k < m; k++)
            {
                MN(k, b) = MN(k, b) / sqrt_n;
                LN(k, b) = LN(k, b) / sqrt_n;
            }

            //printf("kernel called: b=%d, Mn[b]=%f\n",b,Mn[b]);

        }
        else // multiple samples
        {
            int Q = sizeof(pairs) / sizeof(int) / 2;
            double Sg, Sh, lamg, lamh, sigg, sigh, siggh, tmp, sigtau;
            int g,h,ng,nh;
            
                    
            for (int k = 0; k < m; k++)
            {
                for (int s = 0; s < Q; s++)
                {
                
                    g = PAIRS(s, 0);
                    h = PAIRS(s, 1);
                    
                    ng = N[g];
                    nh = N[h];
                    
                    lamg = sqrt( ((double)ng) / (ng + nh) );
                    lamh = sqrt( ((double)nh) / (ng + nh) );

                    for (int i = 0; i < ng + nh; i++)
                    {
                        z[i] = curand_normal_double(localState);
                    }

                    int Xg = 0;
                    for (int t = 0; t < g; t++)
                    {
                        Xg += N[t];
                    }

                    int Xh = 0;
                    for (int t = 0; t < h; t++)
                    {
                        Xh += N[t];
                    }


                    for (int j = 0; j < p; j++)
                    {
                        Sg = 0;
                        for (int i = 0; i < ng; i++)
                        {
                            if(Xstar == nullptr)
                                Sg += z[i] * X[(Xg+i)*p + j];
                            else
                                Sg += z[i] * X[(Xg+Xstar[i])*p + j];    
                        }
                        Sg = Sg / sqrt((double)ng);

                        Sh = 0;
                        for (int i = 0; i < nh; i++)
                        {
                            if(Xstar == nullptr)
                                Sh += z[ng + i] * X[(Xh+i)*p + j];
                            else
                                Sh += z[ng + i] * X[(Xh+Xstar[i])*p + j];
                        }
                        Sh = Sh / sqrt((double)nh);

                        sigg = SIGMA(g,j);
                        sigh = SIGMA(h,j);
                        siggh = sqrt(pow(lamg, 2) * pow(sigg, 2) + pow(lamh, 2) * pow(sigh, 2));

                        if (siggh > 0)
                        {
                            sigtau = pow(siggh, tau[k]);
                            tmp = (Sg * lamg - Sh * lamh) / sigtau;

                            if (tmp > MN(k, b)) MN(k, b) = tmp;
                            if (tmp < LN(k, b)) LN(k, b) = tmp;
                        }
                        //printf("kernel called: b=%d, Ln[b]=%f\n",b,Ln[b]);
                        //printf("kernel called: b=%d, Mn[b]=%f\n",b,Mn[b]);
                    }
                    
                }
            }
        }
}


/* Data organization:
 * X: a 1D array of length M*p, where M=sum(N); the first p elements constitute the first observation (vector) of the first sample,
 *    and the first p*N[0] constitute the first sample, etc.
 * G: the number of samples
 * p: the dimension
 * N: a 1D array of length G
 * pairs: a 1D array of length 2*Q, where Q is the number of pairs; the first two elements constistute the first pair, etc.
 * sigma: a 1D array of length G*p, where the first p elements are standard deviation of the first sample, etc
 * tau:   a 1D array of length m, where m is the number of candidate values for tau
 * m:     the number of candidate values for tau
 * B:     the number of bootstrap replicates
 * state: seed for RNG
 * Mn:    output variable, the unsorted Mn statistics; must be pre-allocated of length B*m
 * Ln:    output variable, the unsorted Ln statistics; must be pre-allocated of length B*m
 */
__global__
void bootstrap_kernel(double* X, int G, int p, int* N, int* pairs, double* sigma,
    double* tau, int m, int B, curandStateMRG32k3a* state,
    double* Mn, double* Ln, double* Z, int Z_stride) {

    int T = blockDim.x;

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    //printf("nBlock=%d,nThreadPerBlock=%d\n",blockDim.x,gridDim.x);
    
    curandStateMRG32k3a localState = state[id]; // for efficiency
    
    for (int b = id; b < B; b += T * gridDim.x)
    {
         bootstrap_dev(X, G, p, N, pairs, sigma,
            tau, m, B, b, &localState, Mn, Ln, nullptr, Z+id*Z_stride);
    }
}

int compare(const void* a, const void* b)
{
    double x = *((double*)a);
    double y = *((double*)b);

    if (x == y) return 0;
    else if (x < y) return -1;
    else return 1;
}

void pvalue(const NumericMatrix& vX,
    const NumericVector& vN,
    const NumericMatrix& vsigma,
    const NumericMatrix& vmu,
    const NumericVector& vtau,
    int* pairs,
    int side,
    int Q,
    int B,
    double* Mn, double* Ln, double* pval)
{
    int G = vN.size();
    int m = vtau.size();
    int p = vsigma.ncol();

    if (G == 1) // one-sample
    {
        int n = vN[0];

        double sig, tmp, eta, eta1, eta2, zl, zu;
        double sqrtn = sqrt((double)n);

        for (int k = 0; k < m; k++)
        {

            zl = std::numeric_limits<double>::max();
            zu = std::numeric_limits<double>::min();

            // compute zl and zu corresponding to the R version
            for (int j = 0; j < p; j++)
            {
                sig = pow(vsigma(0, j), vtau(k));
                if (sig != 0.0)
                {
                    tmp = vmu(0, j) * sqrtn / sig;
                    zl = tmp < zl ? tmp : zl;
                    zu = tmp > zu ? tmp : zu;
                }
            }

            /* DEBUG
            printf("zl=%f, zu=%f\n",zl,zu);
            */

            // compute eta

            eta1 = 0;
            eta2 = 0;
            for (int b = 0; b < B; b++)
            {
                //if(Mn[b+B*k] >= zu) eta1 = eta1 + 1.0;
                //if(Ln[b+B*k] <= zl) eta2 = eta2 + 1.0;

                if (MN(k, b) >= zu) eta1 += 1.0;
                if (LN(k, b) <= zl) eta2 += 1.0;
            }

            eta1 = eta1 / B;
            eta2 = eta2 / B;
            eta = eta1 < eta2 ? eta1 : eta2;

            /* DEBUG
            printf("eta1=%f, eta2=%f, eta=%f\n",eta1,eta2,eta);
            */
            
            if(side == 0)
                pval[k] = 2 * eta < 1 ? 2 * eta : 1;
            else if(side == -1)
                pval[k] = eta1;
            else
                pval[k] = eta2;
                
            //eta.lower <- mean(Mn.sorted[[v]] >= zu)
            //eta.upper <- mean(Ln.sorted[[v]] <= zl)
            //eta.both <- min(1,2*min(eta.lower,eta.upper))
            
            //switch(side,'both' = eta.both,
            //       'lower' = eta.lower,
            //       'upper' = eta.upper)

        }
    }
    else // multiple sample
    {
        
        double sig, tmp, eta, eta1, eta2, zl, zu;
        double ng,nh;
        int g,h;
        
        for (int k = 0; k < m; k++)
        {
        
            zl = std::numeric_limits<double>::max();
            zu = std::numeric_limits<double>::min();
            
            for (int s = 0; s < Q; s++)
            {
                
                g = PAIRS(s, 0);
                h = PAIRS(s, 1);
                
                ng = vN[g];
                nh = vN[h];
                double n = ng * nh / (ng + nh);
                double sqrtn = sqrt(n);
                double lamg = sqrt(ng / (ng + nh));
                double lamh = sqrt(nh / (ng + nh));
                
                
                // compute zl and zu corresponding to the R version
                for (int j = 0; j < p; j++)
                {
                    double sig_tmp = sqrt(pow(lamg, 2) * pow(vsigma(g, j), 2) + pow(lamh, 2) * pow(vsigma(h, j), 2));
                    sig = pow(sig_tmp, vtau(k));

                    if (sig != 0.0)
                    {
                        tmp = (vmu(g, j) - vmu(h, j)) * sqrtn / sig;
                        zl = tmp < zl ? tmp : zl;
                        zu = tmp > zu ? tmp : zu;
                    }
                                //printf("vmu(PAIRS(s, 0), j)=%f, vmu(PAIRS(s, 1), j)=%f\n",vmu(PAIRS(s, 0), j),vmu(PAIRS(s, 1), j));
                                //printf("PAIRS(s, 0)=%d, PAIRS(s, 1)=%d\n",PAIRS(s, 0),PAIRS(s, 1));
                                //printf("j=%d, s=%d\n", j, s);
                }
                
            }

            // DEBUG
            //printf("zl=%f, zu=%f\n",zl,zu);
            //printf("tmp=%f\n",tmp);



            // compute eta

            eta1 = 0;
            eta2 = 0;
            for (int b = 0; b < B; b++)
            {

                if (MN(k, b) >= zu) eta1 += 1.0;
                if (LN(k, b) <= zl) eta2 += 1.0;
            }

            eta1 = eta1 / B;
            eta2 = eta2 / B;
            eta = eta1 < eta2 ? eta1 : eta2;

            // DEBUG
            //printf("eta1=%f, eta2=%f, eta=%f\n",eta1,eta2,eta);

            if(side == 0)
                pval[k] = 2 * eta < 1 ? 2 * eta : 1;
            else if(side == -1)
                pval[k] = eta1;
            else
                pval[k] = eta2;

            //pval[k] = 2 * eta < 1 ? 2 * eta : 1;
        }
        
    }
}

// see size_tau_kernel for the description of arguments
__device__
void pvalue_dev(double* X, double* z, int G, int p, int* N, 
    int* pairs, int side, double* sigma,
    double* tau, int m, int B,
    double* Mn, double* Ln, int r,
    double* pval, double* mu, int* Xstar)
{
    if (G == 1)
    {
        int n = N[0];
        double sig, tmp, zl, zu, eta1, eta2, eta;
        double sqrtn = sqrt(double(n));


        /* compute mean of X */
        for (int j = 0; j < p; j++)
        {
            MU(r, j) = 0;
            for (int i = 0; i < n; i++)
            {
                if(Xstar == nullptr)
                    MU(r, j) += z[i] * X[j + i * p];
                else
                    MU(r, j) += z[i] * X[Xstar[i]*p + j];
            }

            MU(r, j) = MU(r, j) / n;  // conditional on X, mu now follows the Gaussian(0,S/n), where S is sample cov of X
        }

        for (int k = 0; k < m; k++)
        {

            zl = POSITIVE_INF; //__longlong_as_double(0x7ff0000000000000ULL); // Inf
            zu = NEGATIVE_INF; // __longlong_as_double(0xfff0000000000000); // -Inf

            // compute zl and zu corresponding to the R version
            for (int j = 0; j < p; j++)
            {
                sig = pow(sigma[j], tau[k]);
                if (sig != 0.0)
                {
                    tmp = MU(r, j) * sqrtn / sig;
                    zl = tmp < zl ? tmp : zl;
                    zu = tmp > zu ? tmp : zu;
                }
            }

            // compute eta

            eta1 = 0;
            eta2 = 0;
            for (int b = 0; b < B; b++)
            {
                if (MN(k, b) >= zu) eta1 += 1.0;
                if (LN(k, b) <= zl) eta2 += 1.0;
            }

            eta1 = eta1 / B;
            eta2 = eta2 / B;
            eta = eta1 < eta2 ? eta1 : eta2;

            //pval[k + r * m] = 2 * eta < 1 ? 2 * eta : 1;
            
            if(side == 0)
                pval[k + r * m] = 2 * eta < 1 ? 2 * eta : 1;
            else if(side == -1)
                pval[k + r * m] = eta1;
            else
                pval[k + r * m] = eta2;

        }

    }
    else // multiple samples
    {
        double sig, tmp, zl, zu, eta1, eta2, eta;
        
        /*compute the number of observations*/
        int total_n = 0;
        for (int g = 0; g < G; g++) total_n += N[g];

        /* compute mean of X */
        int ghead = 0;

        for (int g = 0; g < G; g++)
        {
            /*compute the number of mean values we have already obtained*/

            for (int j = 0; j < p; j++)
            {
                MU((r*G+g),j) = 0; 
                for (int i = 0; i < N[g]; i++)
                {
                    if(Xstar == nullptr)
                        MU((r*G+g),j) += z[ghead+i] * X[j + (ghead+i) * p];
                    else
                        MU((r*G+g),j) += z[ghead+i] * X[j + (ghead+Xstar[ghead+i]) * p];
                }
                MU((r*G+g),j) = MU((r*G+g),j) / N[g]; // conditional on X, mu now follows the Gaussian(0,S/n), where S is sample cov of X
            }
            
            ghead += N[g];
        }
        
        for (int k = 0; k < m; k++)
        {

                zl = POSITIVE_INF; //__longlong_as_double(0x7ff0000000000000ULL); // Inf
                zu = NEGATIVE_INF; // __longlong_as_double(0xfff0000000000000); // -Inf


                // compute zl and zu corresponding to the R version
                
                double ng, nh, lamg, lamh, sqrtn, sigg, sigh, siggh;
                int g,h;

                for (int s = 0; s < sizeof(pairs) / sizeof(int) / 2; s++)  // for each pair
                {
                    g = PAIRS(s,0);
                    h = PAIRS(s,1);
                    
                    ng = N[g];
                    nh = N[h];
                    
                    lamg = sqrt( ng / (ng+nh) );
                    lamh = sqrt( nh / (ng+nh) );

                    sqrtn = sqrt( ng * nh / (ng + nh) );

                    for (int j = 0; j < p; j++)
                    {
                        sigg = SIGMA(g,j);
                        sigh = SIGMA(h,j);
                        siggh = sqrt(pow(sigg, 2) * pow(lamg,2) + pow(sigh, 2) *pow(lamh,2));
                        sig = pow(siggh, tau[k]);

                        if (sig != 0.0)
                        {
                            tmp = (MU((r*G+g),j) - MU((r*G+h),j) ) * sqrtn / sig;
                            zl = tmp < zl ? tmp : zl;
                            zu = tmp > zu ? tmp : zu;
                        }
                    }

                }
                
                // compute eta

                eta1 = 0;
                eta2 = 0;
                for (int b = 0; b < B; b++)
                {
                    if (MN(k, b) >= zu) eta1 += 1.0;
                    if (LN(k, b) <= zl) eta2 += 1.0;
                }

                eta1 = eta1 / B;
                eta2 = eta2 / B;
                eta = eta1 < eta2 ? eta1 : eta2;

                //pval[k + r * m] = 2 * eta < 1 ? 2 * eta : 1;
                
                if(side == 0)
                    pval[k + r * m] = 2 * eta < 1 ? 2 * eta : 1;
                else if(side == -1)
                    pval[k + r * m] = eta1;
                else
                    pval[k + r * m] = eta2;

        }

    }

}


/* X: data
 * G: number of groups
 * p: dimension
 * N: samples for each group
 * pairs: pairs for test
 * sigma: SIGMA(g,j) is the standard deviation of the j-th coordinate of the g-th group
 * tau: candidate values of the decaying parameter
 * m:  the number of candidate values of tau
 * B: the number of bootstrap replicates
 * curandStateMRG32k3a: for seeding
 * Mn: the empirical distribution of the max statistics
 * Ln: the empirical distribution of the min statistics
 * method: method for selecting tau
 * pval: to be computed in this function; pval[r*m+k] is the p-value of the r-th replicate with tau[k]
 * R: number of Monte Carlo replicates
 * mu: a G*p*R temporary array to be used by the function pvalue_dev
*/
__global__
void size_tau_kernel(double* X, int G, int p, int* N, int* pairs, int side, double* sigma,
    double* tau, int m, int B, curandStateMRG32k3a* state,
    double* Mn, double* Ln, int method, double* size, int R, double* mu, double* Z, int Z_stride, int* Xstar)
{

    int T = blockDim.x;

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= R) return;

    //printf("nBlock=%d,nThreadPerBlock=%d\n",blockDim.x,gridDim.x);


    curandStateMRG32k3a localState = state[id]; // for efficiency


    int total_n = 0;
    for (int g = 0; g < G; g++) total_n += N[g];

    double* z = Z + id*Z_stride; //new double[total_n];

    for (int r = id; r < R; r += T * gridDim.x)
    {
        if (method == 0) // MGB or MGBA
        {
                for (int i = 0; i < total_n; i++)
                {
                    z[i] = curand_normal_double(&localState);
                }
                pvalue_dev(X, z, G, p, N, pairs, side, sigma, tau, m, B, Mn, Ln, r, size, mu, nullptr);
        }
        else //if(method > 0) // RMGB or RMGBA or WB or WBA
        {
            //printf("RMGB called, method=%d\n",method);
        
            for (int r = id; r < R; r += T * gridDim.x)
            {
                // temporary memory to be used by this thread
                int* local_Xstar = Xstar + id*total_n;
                double *local_sigma = sigma + (id*G)*p;
                double *local_mu = mu+(id*G)*p;
                int cumN = 0;
                
                // resample data here
                for(int g=0; g < G; g++)
                {
                    for(int i=0; i < N[g]; i++)
                        local_Xstar[cumN + i] = ceilf(curand_uniform(&localState) * N[g]) - 1; // random integer between 0 and N[g]-1 (inclusive)
                    
                    cumN += N[g];
                }
     
                // compute sigma and mu here
                double u = 0, u2=0;
                cumN = 0;
                double tmp = 0;
                for(int g=0; g < G; g++)
                {
                    for(int j=0; j <p; j++)
                    {
                        u = 0;
                        u2 = 0;
                        for(int i = 0; i < N[g]; i++)
                        {
                            tmp = X[(cumN + local_Xstar[cumN+i])*p + j];
                            u += tmp;
                            u2 += pow(tmp,2);
                        }
                        
                        //mu[(id*G+g)*p + j] = u/N[g];
                        local_sigma[g*p + j] = sqrt((u2-pow(u,2)) / N[g]);
                    }
                    
                    cumN += N[g];
                }
                
                // re-compute Mn and Ln here
                double *local_Mn, *local_Ln;
                local_Mn = Mn + id*m*B;
                local_Ln = Ln + id*m*B;
                
                for(int b=0; b <B; b++)
                    bootstrap_dev(X, G, p, N, pairs, local_sigma,
                        tau, m, B, b, state+id, local_Mn, local_Ln, local_Xstar,Z+id*Z_stride);
                        
                if(method ==  1) // RMGB or RMGBA
                {
                    // generate Gaussian variables
                    for (int i = 0; i < total_n; i++)
                    {
                        z[i] = curand_normal_double(&localState);
                    }
                }
                else
                {
                    for (int i = 0; i < total_n; i++)
                    {
                        z[i] = 1;
                    }
                }
                
                pvalue_dev(X, z, G, p, N, pairs, side, local_sigma, tau, m, B, local_Mn, local_Ln, 0, size+r*m, local_mu, local_Xstar);
                //pvalue_dev(X, z, G, p, N, pairs, sigma,       tau, m, B, Mn,       Ln,       r, size,     mu);
            }
        }
    }

    // clean up:
    // delete z;
}

/* vX: data
 * vN: samples for each group
 * vpairs_t: pairs for test
 * vsigma: SIGMA(g,j) is the standard deviation of the j-th coordinate of the g-th group
 * vtau: candidate values of the decaying parameter
 * B: the number of bootstrap replicates
 * curandStateMRG32k3a: for seeding
 * alpha: significance level
 * method: method for selecting tau
 * R: number of Monte Carlo replicates for choosing tau
 * seed: an integer to seed random number generator
 * vpval: to be computed in this function; the p-value for each tau
 * vsize: to be computed in this function: the estimated size for each tau
 * nblock: the number of blocks
 * threads_per_block: number of threads per block
*/
extern "C"
int anova_cuda_(const NumericMatrix & vX,
    const NumericVector & vN,
    const NumericMatrix & vsigma,
    const NumericMatrix & vmu,
    const NumericVector & vtau,
    Nullable<NumericMatrix> vpairs_t,
    int side,
    int B,
    double alpha,
    int method,
    int R,
    int nblock,
    int threads_per_block,
    int seed,
    NumericVector & vpval,
    NumericVector & vsize,
    NumericMatrix & vMn,
    NumericMatrix & vLn)
{

    int G = vN.size();
    int m = vtau.size();
    int p = vsigma.ncol();

    int totalThreads = threads_per_block * nblock;

    int* pairs;
    int* N;
    double* tau;
    double* sigma;
    double* X;
    double* Mn;
    double* Ln;
    curandStateMRG32k3a* devMRGStates;
    
    int Q = 0;
    double* Z = nullptr;
    int Z_stride = 0;
    int total_N = 0;
    int n = 0;
    double* pval = nullptr;
    double* size = nullptr;
    int *Xstar = nullptr;
    double* mu = nullptr;

    /* initialize N */
    bool msuccess = gpuAssert(cudaMallocManaged(&N, G * sizeof(int)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    
    
    for (int g = 0; g < G; g++) {
        N[g] = vN[g];
        total_N += N[g];
    }

    /* DEBUG */
    //printf("G=%d, p=%d, m=%d, B=%d, total_N=%d\n",G,p,m,B,total_N);


    /* allocate shared memory for X, tau, sigma, Mn, Ln, devMRGStates */
    msuccess = gpuAssert(cudaMallocManaged(&X, total_N * p * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    msuccess = gpuAssert(cudaMallocManaged(&tau, m * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    msuccess = gpuAssert(cudaMallocManaged(&sigma, G * p * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    msuccess = gpuAssert(cudaMallocManaged(&Mn, m * B * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    msuccess = gpuAssert(cudaMallocManaged(&Ln, m * B * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    
    //GPU_ERR_CHK(cudaMallocManaged(&X, total_N * p * sizeof(double)));
    //GPU_ERR_CHK(cudaMallocManaged(&tau, m * sizeof(double)));
    //GPU_ERR_CHK(cudaMallocManaged(&sigma, G * p * sizeof(double)));
    //GPU_ERR_CHK(cudaMallocManaged(&Mn, m * B * sizeof(double)));
    //GPU_ERR_CHK(cudaMallocManaged(&Ln, m * B * sizeof(double)));
    //if(vpairs.ncol()==2)
    if (vpairs_t.isNotNull())
    {
        NumericMatrix vpairs(vpairs_t.get());
        //GPU_ERR_CHK(cudaMallocManaged(&pairs, vpairs.nrow() * vpairs.ncol() * sizeof(double)));
        msuccess = gpuAssert(cudaMallocManaged(&pairs, vpairs.nrow() * vpairs.ncol() * sizeof(double)), __FILE__, __LINE__);
        if(msuccess == false) goto cleanup;
    
    }

    msuccess = gpuAssert(cudaMalloc((void**)&devMRGStates, totalThreads * sizeof(curandStateMRG32k3a)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    //GPU_ERR_CHK(cudaMalloc((void**)&devMRGStates, totalThreads * sizeof(curandStateMRG32k3a)));
    
    setup_kernel << <nblock, threads_per_block >> > (devMRGStates, seed);


    /* initialize X with centered data */
    
    for (int g = 0; g < G; g++)
    {
        for (int i = 0; i < N[g]; i++)
        {
            for (int j = 0; j < p; j++)
                X[n * p + j] = vX(n, j) - vmu(g, j);

            n += 1;
            //printf("X(%d,0)=%f, vmu(%d,0)=%f, vX(%d,0)=%f\n",i,X[i*p], g, vmu(g,0), i, vX(i,0));
        }
    }

    /* initialize sigma and tau */
    for (int g = 0; g < G; g++)
        for (int j = 0; j < p; j++) SIGMA(g, j) = vsigma(g, j);

    for (int k = 0; k < m; k++) tau[k] = vtau[k];


    
    if (vpairs_t.isNotNull())
    {
        NumericMatrix vpairs(vpairs_t.get());
        Q = vpairs.nrow();
        for (int s = 0; s < Q; s++)
        {
            PAIRS(s, 0) = vpairs(s, 0) - 1;
            PAIRS(s, 1) = vpairs(s, 1) - 1;
        }
    }
    
    
    
    // pre-allocate memory for Z
    if (G == 1) // one-sample
    {
        Z_stride = N[0];
    }
    else
    {
            int g,h;
            for (int s = 0; s < Q; s++)
            {
                g = PAIRS(s, 0);
                h = PAIRS(s, 1);
                if(N[g] + N[h] > Z_stride) Z_stride = N[g] + N[h];
            }
            
    }
    
    //GPU_ERR_CHK(cudaMallocManaged(&Z, Z_stride * min(B,totalThreads) * sizeof(double)));
    msuccess = gpuAssert(cudaMallocManaged(&Z, Z_stride * min(B,totalThreads) * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    

    /* compute the empirical distribution of Mn and Ln */
    bootstrap_kernel << <nblock, threads_per_block >> > (X, G, p, N, pairs, sigma,
        tau, m, B, devMRGStates, Mn, Ln, Z, Z_stride);

    cudaDeviceSynchronize(); //Wait for GPU to finish executing kernel.


    /* sort Mn and Ln */
    for (int k = 0; k < m; k++)
    {
        qsort(Mn + k * B, B, sizeof(double), compare);
        qsort(Ln + k * B, B, sizeof(double), compare);
    }
    
    /* copy Mn and Ln to vMn and vLn */
    for(int k = 0; k < m; k++)
    {
        for(int b=0; b < B; b++)
        {
            vMn(k,b) = Mn[k*B+b];
            vLn(k,b) = Ln[k*B+b];
        }
    }

    /* DEBUG  */
    /*int q = m > 1 ? 2 : 1;
    for(int b=0; b < q*B; b++)
        if(b % 100 == 0) printf("Mn[%d]=%f\n",b,Mn[b]);
    for(int b=0; b < q*B; b++)
        if(b % 100 == 0) printf("Ln[%d]=%f\n",b,Ln[b]);
    */

    /* compute p-value */

    //pval = new double[m];
    msuccess = gpuAssert(cudaMallocManaged(&pval, m * sizeof(double)), __FILE__, __LINE__);
    if(msuccess == false) goto cleanup;
    

    pvalue(vX, vN, vsigma, vmu, vtau, pairs, side, Q, B , Mn, Ln, pval);
    for (int k = 0; k < m; k++) vpval[k] = pval[k];

    //for(int k = 0; k<m; k++) printf("pvalue with tau=%f is %f\n",vtau[k],pval[k]);

    /*
    for(int b=0; b < q*B; b++)
        if(b % 100 == 0) printf("Mn[%d]=%f\n",b,Mn[b]);
    for(int b=0; b < q*B; b++)
        if(b % 100 == 0) printf("Ln[%d]=%f\n",b,Ln[b]);
    */

    /* select tau when R > 0*/

    if (R > 0)
    {
        
        msuccess = gpuAssert(cudaMallocManaged(&size, m * R * sizeof(double)), __FILE__, __LINE__);
        if(msuccess == false) goto cleanup;
    
        //GPU_ERR_CHK( cudaMallocManaged(&size, m * R * sizeof(double)) );

        cudaFree(Z);
        Z_stride = total_N;
        msuccess = gpuAssert(cudaMallocManaged(&Z, Z_stride * B * sizeof(double)), __FILE__, __LINE__);
        if(msuccess == false) goto cleanup;
    
        //GPU_ERR_CHK(cudaMallocManaged(&Z, Z_stride * B * sizeof(double)));
        
        
        
        if(method > 0) // 'RMGB', 'RMGBA', 'WB', 'WBA'
        {
            cudaFree(sigma);
            cudaFree(Mn);
            cudaFree(Ln);
            
            int R0 = min(R,totalThreads);
            
            msuccess = gpuAssert(cudaMallocManaged(&Xstar, R0 * total_N * sizeof(int)), __FILE__, __LINE__);
            if(msuccess == false) goto cleanup;
            msuccess = gpuAssert(cudaMallocManaged(&sigma, R0 * G * p * sizeof(double)), __FILE__, __LINE__);
            if(msuccess == false) goto cleanup;
            msuccess = gpuAssert(cudaMallocManaged(&Mn, R0 * m * B * sizeof(double)), __FILE__, __LINE__);
            if(msuccess == false) goto cleanup;
            msuccess = gpuAssert(cudaMallocManaged(&Ln, R0 * m * B * sizeof(double)), __FILE__, __LINE__);
            if(msuccess == false) goto cleanup;
            msuccess = gpuAssert(cudaMallocManaged(&mu, R0 * G * p * sizeof(double)), __FILE__, __LINE__);
            if(msuccess == false) goto cleanup;
    
        
            //GPU_ERR_CHK( cudaMallocManaged(&Xstar, R0 * total_N * sizeof(int)) );
            //GPU_ERR_CHK( cudaMallocManaged(&sigma, R0 * G * p * sizeof(double)) );
            //GPU_ERR_CHK( cudaMallocManaged(&Mn, R0 * m * B * sizeof(double)) );
            //GPU_ERR_CHK( cudaMallocManaged(&Ln, R0 * m * B * sizeof(double)) );
            
            //GPU_ERR_CHK( cudaMallocManaged(&mu, R0 * G * p * sizeof(double)) );
            
        }
        else
        {
            msuccess = gpuAssert(cudaMallocManaged(&mu, p * R * G * sizeof(double)), __FILE__, __LINE__);
            if(msuccess == false) goto cleanup;
            //GPU_ERR_CHK( cudaMallocManaged(&mu, p * R * G * sizeof(double)) );
        }


        size_tau_kernel << <nblock, threads_per_block >> > (X, G, p, N, pairs, side, sigma,
            tau, m, B, devMRGStates, Mn, Ln, method, size, R, mu, Z, Z_stride, Xstar);

        cudaDeviceSynchronize(); //Wait for GPU to finish executing kernel.

        for (int k = 0; k < m; k++)
        {

            vsize[k] = 0.0;
            for (int r = 0; r < R; r++)
            {
                //printf("k=%d,size[k,r]=%f\n",k,size[k+r*m]);
                if (size[k + r * m] < alpha)
                    vsize[k] = vsize[k] + 1;
            }
            vsize[k] = vsize[k] / R;
        }

        
    }

    /* DEBUG
    int q = m > 1 ? 2 : 1;
    for(int b=0; b < q*B; b++)
        if(b % 50 == 0) printf("Mn[%d]=%f\n",b,Mn[b]);
    for(int b=0; b < q*B; b++)
        if(b % 50 == 0) printf("Ln[%d]=%f\n",b,Ln[b]);
    */

    // Free memory
cleanup:
    cudaFree(N);
    cudaFree(X);
    cudaFree(tau);
    cudaFree(sigma);
    cudaFree(Mn);
    cudaFree(Ln);
    cudaFree(devMRGStates);
    cudaFree(Z);
    cudaFree(size);
    cudaFree(mu);
    cudaFree(Xstar);
    cudaFree(pval);

    if (vpairs_t.isNotNull())  cudaFree(pairs);

    if(msuccess) return 0;
    else return 1;
}
