# CommunityDetection

In this repository, we present two different algorithms that solve the Community Detection Problem by maximizing the modularity. In addition, we include all the tests that have been performed to compare the quality of the algorithms. This repository and its content has been created by Xabier Benavides and Unai Lizarralde.

# Algorithms

1. <b>Iterated Variable Neighborhood Search</b> with a modified modularity computation. (IVNS)

2. <b>Univariate Marginal Distribution</b> algorithm that uses a locus-based-adjacency encoding to estimate the marginal probabilities. (UMDA)

# Tests

- <b>Instance</b>: NIPS papers database (2014-2015). The task is to cluster the authors according to the papers that have published jointly with other authors.
  
- <b>Conditions</b>:

  - <b>Maximum number of evaluations</b>: 2000 and 20.000.

  - <b>Number of communities</b>: From 2 to 100.
  
- <b>Value of the hyperparameters</b>:
  
  - <b>IVNS</b>:
    - <i>Num_pert</i>: 1
    - <i>Exp_pert</i>: 1
