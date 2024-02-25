library(reticulate)
library(irlba)
library(sinkr)

np <- import("numpy")
Xt <- np$load("dataset.npy")
frac.gaps <- 0.5

# The dineof "interpolated" field
set.seed(1)
RES <- dineof(Xt, delta.rms = 1e-03) # lower 'delta.rms' for higher resolved interpolation
Xa <- RES$Xa

tfile <- tempfile(fileext=".npy")
np$save(tfile, Xa)
