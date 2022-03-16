## overwiev
[Wikipedia](https://en.wikipedia.org/wiki/Time_series)  
--> **Time Domain**:  

---
### **Probably** useful
#### cross-correlation
[wiki:](https://en.wikipedia.org/wiki/Cross-correlation)  
is a measure of similarity of two series as a function of the displacement of one relative to the other


--- 
### **Maybe** useful
#### Multivariate: canonical correlation
[wiki](https://en.wikipedia.org/wiki/Canonical_correlation).  like Linear regression but also works in the sace of multiple intercorrelated outcome variables. I.e.:  
(yield, biomass, ...) ~ spectral bands(for each data) + spectral indicies(for each date) ...   

#### Breusch–Godfrey test
[wiki](https://en.wikipedia.org/wiki/Breusch%E2%80%93Godfrey_test)  
test is used to assess the validity of some of the modelling assumptions inherent in applying regression-like models to observed data series.[1][2] In particular, it tests for the presence of serial correlation that has not been included in a proposed model structure and which, if present, would mean that incorrect conclusions would be drawn from other tests or that sub-optimal estimates of model parameters would be obtained. 

#### VAR - Vector autoregression
[wiki](https://en.wikipedia.org/wiki/Vector_autoregression):  
Multidimensional autoregressive model

---
### **NOT** useful
#### Decomposition of TS
[wiki](https://en.wikipedia.org/wiki/Decomposition_of_time_series) 
decomposition of random and non-random parts

#### Seasonal adjustment
remove seasonal effects to study "local" effects

#### Granger Causality
predict one TS with another TS

#### Dickey-Fuller test
checks some irrelevant property (checks if stochastic process has uniroot - if not, it is non-stationalry and may have no trend)

#### Johansen test
testing coointegration

#### Ljung–Box test
test "randomness" of whole TS

#### Durbin–Watson statistic
tests presence of autocorrelation (at lag 1). 
--> prove that 

#### ARMA - Autoregressive–moving-average model
[wiki:](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)  
mix of autoregressive and movin average  
$X_{t}=c+\varepsilon_{t}+\sum_{i=1}^{p} \varphi_{i} X_{t-i}+\sum_{i=1}^{q} \theta_{i} \varepsilon_{t-i}$

#### ARCH - Autoregressive conditional heteroskedasticity
[wiki:](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)  
TS which integrates heteroskedasiticity: $var(\epsilon_{t}) = f(\epsilon_{t-1}, \epsilon_{t-2}...\epsilon_{t-p})

