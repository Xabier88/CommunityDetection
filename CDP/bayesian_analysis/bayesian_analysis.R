#First, set working directory to current directory.
setwd(".")

#Remove previous variables.
rm(list = ls())

#Read the csv files
results1<-read.csv(file="../csv/one_alg_test_statistics.csv",header=TRUE,sep=",",stringsAsFactor=FALSE)
results2<-read.csv(file="../csv/pop_alg_test_statistics.csv",header=TRUE,sep=",",stringsAsFactor=FALSE)

#Multiply average fitness by -1 (From maximization to minimization)
for (i in 1:length(results1$Fitness_average)) 
{
  results1$Fitness_average[i] <- -results1$Fitness_average[i]
  results2$Fitness_average[i] <- -results2$Fitness_average[i]
}

#Import packages.
source("bayesian.R")
library("scmamp")
library("ggplot2")
library(latex2exp)

#Compute and sample posterior probabilities.
test.results <- bSignedRankTest(x=results1$Fitness_average, y=results2$Fitness_average, rope=c(-0.05, 0.05))
print(test.results$posterior)
colMeans(test.results$posterior)
test.results$posterior.probabilities

#Store results in a Simplex plot.
p1<-plotSimplex(test.results, plot.density=TRUE, A="UMDA",B="IVNS", plot.points=TRUE, posterior.label=FALSE, alpha=0.5, point.size=5 ,font.size = 5)
ggsave("Simplex.svg",p1, width = 125, height = 125, dpi = 300, units = "mm", device='svg')
