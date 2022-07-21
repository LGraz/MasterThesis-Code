require <- function(package){
    package <- as.character(substitute(package))
    if (package %in% rownames(installed.packages())){
        base::library(package, character.only = TRUE)
    } else {
        cat("install package: ",package,"\n")
        install.packages(package)
        base::library(package, character.only = TRUE)
    }
}
library <- require

