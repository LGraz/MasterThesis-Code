# define functions which will be used as covariates.
# i.e.   covariates = c(max(gdd, ts), min(gdd, ts), ...)

ndvi_min <- 0.3


# approximate integral of ts over (0, gdd) by linear-cumsum
integral_up_to_gdd <- function(gdd, ndvi_min. = ndvi_min) {
    function(x, y) {
        stopifnot(
            is.numeric(x), is.numeric(y),
            length(x) == length(y), length(x) > 2
        )
        n <- max(x) - min(x) + 1
        lin_interpol <- approx(x, y, n = n)
        if (!(gdd %in% lin_interpol$x)) {
            stop(paste(
                "gdd not in range of data, range: ",
                range(lin_interpol$x), "  gdd:", gdd
            ))
            a <- abs(gdd - max(x))
            b <- abs(gdd - min(x))
            gdd <- if (a > b) 1 else max(x)
        }
        yy <- sapply(lin_interpol$y - ndvi_min., max, 0)
        # print(x)
        # print(gdd)
        # print(which(x <= gdd))
        csum <- cumsum(yy[1:max(which(x <= gdd))])
        csum[length(csum)]
    }
}

integral_over_all <- function(x, y) {
    integral_up_to_gdd(max(x))(x, y)
}

integral_up_to_peak <- function(x, y) {
    gdd_peak <- which.max(y)
    integral_up_to_gdd(x[gdd_peak])(x, y)
}

integral_after_peak <- function(x, y) {
    gdd_peak <- which.max(y)
    # cat("###", gdd_peak, "\n\n\n", str(gdd_peak), "__", length(gdd_peak))
    stopifnot(length(gdd_peak) == 1)
    integral_over_all(x, y) - integral_up_to_gdd(x[gdd_peak])(x, y)
}

get_max_slope <- function(x, y) {
    stopifnot(
        is.numeric(x), is.numeric(y),
        length(x) == length(y), length(x) > 2
    )
    n <- max(x) - min(x) + 1
    lin_interpol <- approx(x, y, n = n)
    n <- length(x)
    diff <- lin_interpol$y[2:n] - lin_interpol$y[1:(n - 1)]
    max(diff)
}

get_min_slope <- function(x, y) {
    stopifnot(
        is.numeric(x), is.numeric(y),
        length(x) == length(y), length(x) > 2
    )
    n <- max(x) - min(x) + 1
    lin_interpol <- approx(x, y, n = n)
    n <- length(x)
    diff <- lin_interpol$y[2:n] - lin_interpol$y[1:(n - 1)]
    min(diff)
}



fun_to_get_covariates <- list(
    min_slope = get_min_slope,
    max_slope = get_max_slope,
    int_all = integral_over_all,
    int_before_peak = integral_up_to_peak,
    int_after_peak = integral_after_peak,
    int_till_tillering = integral_up_to_gdd(685),
    int_from_tillering_to_flowering = function(x, y) {
        integral_up_to_gdd(1075)(x, y) - integral_up_to_gdd(685)(x, y)
    },
    # min = function(x, y) min(y),
    peak = function(x, y) max(y),
    peak_gdd = function(x, y) x[which.max(y)]
)

## from https://ipad.fas.usda.gov/cropexplorer/description.aspx?legendid=313&regionid=na
# 9.0-9.9 	0.1-1.0 	Germination 	70 	    70
# 1.0-2.0 	1.0-1.3 	Emergence1. 	330 	400
# 2.0-2.5 	1.3-2.0 	Tillering 	    285 	685
# 2.5-3.0 	2.0-3.0 	Stem
# 3.0-4.0 	3.0-3.6 	Booting 	    190 	875
# 4.0-5.0 	3.6-4.0 	Flowering 	    200 	1075
# 5.0-5.3 	4.0-4.5 	Milky
# 5.3-6.0 	4.5-5.0 	Waxy Ripe2. 	450-530 1575
