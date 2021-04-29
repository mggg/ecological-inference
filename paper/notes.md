# Notes for draft
seeks to address some limitations of the existing R libraries, namely that:

- different ecological inference methods are spread out across different libraries
- existing implementations may use slower-to-converge MCMC approaches (BENCHMARK)
- existing libraries do not all expose samples directly (CHECK THAT THIS IS STILL THE CASE) and do not incorporate convergence-related checks and warnings
- not all libraries are oriented towards careful uncertainty quantification (CHECK AND EXPLAIN/SAY THIS MORE CAREFULLY)

\textcolor{violet}{Notes on R packages}

  - `ei`
    - main function is inplementation of King's EI (truncated normal~1997)... functionality is mainly oriented toward 2x2.  Does include RxC generalization and Goodman ER
    - Can get samples of psi, aggregated betas (at the polity level), and precinct-level betas.  The first two are easily grabed by eiread() function.  the latter doesn't seem to be designed to be grabbed, but the samples can be read.
    - convergence testing/diagnostics are not transparent (if they are done at all)... there is a 'resamp' parameter that appears to be some sort of diagnostic, but I'm not sure what it is exactly (not explained much in documentation)
    - standard errors and 80% "confidence intervals" (these appear to be just .1 and .9 percentiles of sample) for precinct-level betas.  Also gives standard errors for psis and aggregate betas.
  - `eiPack`
    - includes multinomial dirichlet (ala Rosen), Goodman ER, ER w/ Bayesian normal regression, and some plotting functionality (density and bounds)
    - outputs draws for alphas, betas, and cell counts as well as acceptance ratios of these variable draws
    - no convergence tests/diagnostics included or described.  Presumably the outputs can make use of the functionality of the coda package to perform these, but user would have to write this script.
    - Has plotting functionality for unit (precinct)-level credible intervals, though this appears to be just for visualization and less for actually grabbing/using these values... no functionality/attention to uncertainty for other measures of interest
  - `eiCompare`
    - Appears to be primarily a wrapper around `ei` and `eiPack` packages.  Some added funcationality (including capturing uncertainty, visualization, and comparison of results across methods).  Some instances appear to be literally rewritten from these packages with minor/trivial tweaks, rather than significantly different implementations.
  - `RxCEcolInf`
    - package really just includes the model from the 2009 Greiner/Quinn paper
    - outputs draws for: internal cell counts, thetas, mu, and the standard deviations and correlations in sigma.
    - convergence tests not directly incorporated, but returns mcmc object with intention of using functionality of R's coda package.  Documentation examples show using coda's Geweke's as well as Heidelberger and Welch's convergence diagnostics, but ackwledges that chains created by `RxCEcolInf` will cause error in coda's Gelman-Rubin diagnostic.
    - the `RxCEcolInf` package itself does not include careful uncertainty quantification functionality, but digging through some of their replication code from their paper does show some examples of uncertainty calculations and credible intervals, though not particularly well documented/clear how to apply