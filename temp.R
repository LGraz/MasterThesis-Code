library(robustbase)
set.seed(1234)
a <- rlnorm(1000)
plot(a)
adjOutlyingness(a)


outl <- adjOutlyingness(a, alpha.cutoff = 0.6)
all(outl$nonOut)
plot(a, col = -1 * outl$nonOut + 2)
