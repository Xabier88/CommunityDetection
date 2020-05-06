# Community Detection

In this repository, we present two different algorithms that solve the Community Detection Problem by maximizing the modularity. In addition, we include the parameter tuning and all the tests that have been performed to compare the quality of the algorithms. This repository and its content has been created by Xabier Benavides and Unai Lizarralde.

## Algorithms

1. <b>Iterated Variable Neighborhood Search</b> with a modified modularity computation. (IVNS)

2. <b>Univariate Marginal Distribution</b> algorithm that uses a locus-based-adjacency encoding to estimate the marginal probabilities. (UMDA)

## Parameter tuning

- <b>Instance</b>: NIPS papers database (2014-2015). The task consists in clustering the authors according to the papers that have published jointly with other authors.
  
- <b>Conditions</b>:

  - <b>Maximum number of evaluations</b>: 20.000.

  - <b>Number of communities</b>: 20.

### IVNS

The hyperparameter optimization has been performed using the hyperopt python package. This package implements a bayesian optimization using the Tree Parzen Estimator (TPE). The hyperparameters that have been optimized are the following:

- <b>Num_pert</b>: Number of communities that are dismantled in the shake procedure. <i>Possible values: Integers from 1 to 10</i>.

- <b>Exp_pert</b>: Exponent that is applied to the size of the communities when computing the probability of dismantling each of them in the shake procedure. <i>Possible values: Real numbers from 0 to 10</i>.

### UMDA

The hyperparameter optimization has been performed using a grid search. The hyperparameters that have been optimized are the following:

- <b>Pop_size</b>: Population size. <i>Possible values: 50, 100, 200, 400</i>.

- <b>Sel_size</b>: Selection size. <i>Possible values: 10, 20, 40, 50</i>.

The remaining parameters have been set by hand:

- <b>Max_mutation</b>: Maximum mutation rate. <i>Value: 0,2</i>.

- <b>Min_mutation</b>: Minimum mutation rate. <i>Value: 0,02</i>.

- <b>Mutation_decrease</b>: Mutation rate decrease factor. <i>Value: 0,02</i>.

## Tests

- <b>Instance</b>: NIPS papers database (2014-2015). The task consists in clustering the authors according to the papers that have published jointly with other authors.
  
- <b>Conditions</b>:

  - <b>Maximum number of evaluations</b>: 2000 and 20.000.

  - <b>Number of communities</b>: From 2 to 100.
