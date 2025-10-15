### Non-negative Matrix Factorization ###
#
# Identifying subnetworks within the full structural connectivity network using 
# nonnegative matrix factorization. This includes cross-validation with data
# imputation. The resulting data are further processed in scripts/results/48_results
#

#install.packages('reticulate')
#devtools::install_github("zdebruine/RcppML")

library(RcppML)
library(reticulate)
library(ggplot2)

np <- import("numpy")

# Read subject SCs data matrix
scs <- t(np$load("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/weights_triu_consensus.npy"))

### Run cross-validation ###
cv_impute <- crossValidate(scs, k = 2:10, method = "impute", reps = 100, seed = 691351275)

# Plot
plot(cv_impute) + ggtitle("impute cross-validation on\nclean dataset")

# Save results
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/cv_impute_mse.npy", cv_impute$value)

# Compute average MSE
avg_cv_impute <- tapply(cv_impute$value, cv_impute$k, mean)

# Plot
ggplot() + geom_point(aes(x = 2:10, y = avg_cv_impute)) + ggtitle("impute cross-validation") #+ ylim(8500, 9500)

# find minimum MSE
min(avg_cv_impute)
which.min(avg_cv_impute)


### Run NMF at determined rank ###
seeds = read.table("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_random_seeds.csv", header = FALSE, sep = "", dec = ".")

model <- nmf(scs, k = 5, seed = seeds[1:100,1])  # select best NMF out of 100 randomly initialized runs
evaluate(model, scs)

# save model
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_components.npy", model@w)
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_weights.npy", model@h)
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_scaler.npy", model@d)


### Run rank-2 NMF ###
seeds = read.table("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_random_seeds.csv", header = FALSE, sep = "", dec = ".")

model_rank2 <- nmf(scs, k = 2, seed = seeds[1:100,1])
evaluate(model_rank2, scs)

#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmfr_components.npy", model_rank2@w)
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmfr_weights.npy", model_rank2@h)
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmfr_scaler.npy", model_rank2@d)


### Build rank-2 to -5 NMF components ###
model_rank3 <- nmf(scs, k = 3, seed = seeds[1:100,1])
model_rank4 <- nmf(scs, k = 4, seed = seeds[1:100,1])
model_rank5 <- nmf(scs, k = 5, seed = seeds[1:100,1])

#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_multirank_factors.npy", cbind(model_rank2@w, model_rank3@w, model_rank4@w, model_rank5@w))


### Run rank-5 NMF with full data matrix ###
scs_full<- t(np$load("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/weights_triu_full.npy"))

model_full <- nmf(scs_full, k = 5, seed = seeds[1:100,1])
evaluate(model_full, scs_full)

#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_full_components.npy", model_full@w)
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_full_weights.npy", model_full@h)
#np$save("/Users/dk/Documents/Charite/PhD/travelingwaves/data/48_results/nmf_full_scaler.npy", model_full@d)