---
title: 'PyEI: A Python package for ecological inference'
tags:
  - Python
  - voting rights
  - ecological inference
  - Bayesian inference
authors:
  - name: Karin Knudson
    orcid: 0000-0003-1312-6473
    affiliation: 
affiliations:
 - name: Data Intensive Studies Center, Tufts University
   index: 1
date: 8 March 2021
bibliography: paper.bib
---

# Summary

An important question in some US voting rights cases and
redistricting litigation is whether and to what degree voting is racially polarized.
In the setting of voting rights cases, ecological inference involves using
observed data about voting outcomes in each of a number of precincts and demographic information
about each precint to infer voting patterns within each demographic group.

More generally, we can think of ecological inference as seeking to use knowledge about the margins of a set of tables (\autoref{fig:table_ex}) to infer associations between the row and column variables, by making (typically probablistic) assumptions about the underlying associations. In the setting of assessing racially polarized voting, each column in a table like the one in \autoref{fig:table_ex} corresponds to a candidate or voting outcome, each row to a racial group, and each table to a precinct. Ecological inference methods then use the known counts of voting outcomes in each precinct and the known counts of people in demographic groups in each precinct to make inferences about the distribution of voting outcomes within each demographic group, thus addressing questions like: "What percentage of voters in Group 1 voted for candidate A?"). The "two by two" ecological inference problem in this example, where we have two groups and two voting outcomes, is a special case of the more general "R by C" ecological inference, in which we may have more than two groups or voting outcomes.
Ecological inference is also applicable in other fields, such as epidemiology and sociology.

![In "two by two" ecological inference we have information about the marginal counts for a set of tables like the one above and would like to make inferences about, for example the number or proportion of members of group 1 who were also in group A.\label{fig:table_ex}](figs/table_ex.png){ width=70% }

# Statement of need

The results of ecological inference for inferring racially polarized voting are used in
US voting rights cases; therefore, easy to use and high quality tools for performing ecological inference are of practical interest. There is a need for an ecological inference library that 
brings together a variety of ecological inference methods in one place and makes easy
crucial tasks such as: quantifying the uncertainty associated with ecological inference
results under a given model; making comparisons between methods; and bringing relevant 
diagnostic tools to bear on ecological inference methods. To address this need, 
we introduce `PyEI`, a Python package for ecological inference. 

`PyEI` is meant to be useful to two main groups of researchers. First, it serves application-oriented researchers and practitioners who seek to run ecological inference on domain data (e.g. voting data), report results, and understand the uncertainty related to those results.
Second, it facilitates exploration and benchmarking for researchers who are seeking to understand properties of existing
ecological inference methods in different settings and/or develop new statistical methods for ecological inference.

`PyEI` brings together the following ecological inference methods in a common framework alongside plotting, reporting, and diagnostic tools:

- Goodman's ecological regression [@goodman1953ecological] and a Bayesian linear regression variant [is there a citation? I'm sure people have done this before...]
- The truncated-normal based method of [@king1997solution] NOT YET IMPLEMENTED
- Binomial-Beta hierarchical models [@king1999binomial]
- Dirichlet-Multinomial hierarchical models [@rosen2001bayesian]
- A Bayesian hierarchical method for 2x2 EI following the approach of @wakefield2004ecological
- A Bayesian hierarchical method for RxC EI following the approach of @greiner2009r NOT YET IMPLEMENTED

Several R libraries for ecological inference exist, such as `ei` [@ei], `eiCompare` [@eiCompare], `eiPack` [@eiPack], and `RxCEcolInf` [@RxCEcolInf]. `PyEI` presents a Python-based option that researchers who primarily use Python may appreciate. It also seeks to address some limitations of the existing R libraries, namely that:

- different ecological inference methods are spread out across different libraries
- existing implementations may use slower-to-converge MCMC approaches (BENCHMARK)
- existing libraries do not all expose samples directly (CHECK THAT THIS IS STILL THE CASE) and do not incorporate convergence-related checks and warnings
- not all libraries are oriented towards careful uncertainty quantification (CHECK AND EXPLAIN/SAY THIS MORE CAREFULLY)

To meet the needs described here, `PyEI` incorporates the following key features and characteristics. First, the Bayesian hierarchical methods implemented in `PyEI` rest on modern probabilistic programming tooling [@salvatier2016probabilistic] and MCMC methods [@hoffman2014no]. Integration with existing tools `PyMC` [@salvatier2016probabilistic] and ArViZ [@arviz_2019] makes the results amenable to state of the art diagnostics (e.g. convergence diagostics) and some reasonable checks are automatically performed. Second, summary and plotting utilities for reporting, visualizing, and comparing results are included (see example plots below), with an emphasis on visualizations and reports that make clear the uncertainty of estimates under a model. Lastly, clear documentation and examples are provided.

# Examples of plotting functionality

![KDE plots for visualing uncertainty of support for candidates within each group.\label{fig:kdes}](figs/kdes.png){ width=60% } 

![Bayesian credible intervals for support of candidates within groups.\label{fig:credible_interval}](figs/credible_interval.png){ width=70% }

![Visualizing and quantifying degree of polarization.\label{fig:polarization}](figs/polarization.png){ width=70% }

![Visualizing estimats and uncertainty for precinct-level estimates.\label{fig:precinct_level}](figs/precinct_level.png){ width=60% }

!["Tomography" plots for two-by-two ecological inference.\label{fig:tomography}](figs/tomography.png){ width=40% }


# Acknowledgements

We acknowledge contributions from ...

# References

