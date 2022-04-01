# Table - overview
parameter: {h: bandwidth, $\lambda$: penalty-factor, k: #neighbors}



## categories
tuning parameter  
unbiased  
asympt. unbiased  
deg. of freedom  
continuous  
smooth ($C^\infty$)  


# Setting
We are given data in the form of $\left(x_{i}, Y_{i}\right)(i=1, \ldots, n)$. Assume that it can be represented by 
$$
Y_{i}=m\left(x_{i}\right)+\varepsilon_{i},
$$
where $\varepsilon_i$ is some noise and $m: \mathbb{R} \rightarrow \mathbb{R}$ being some (non-parametric regression) function. If we assume that $\varepsilon_{1}, \ldots, \varepsilon_{n}$ i.i.d. with $\mathbb{E}\left[\varepsilon_{i}\right]=0$ then $$m(x)=\mathbb{E}[Y \mid x]$$ 
Different assumptions on $m$ will lead to the following models:

# Methods - Description
## Kernel regression
$$
\mathbb{E}[Y \mid X=x]
= \int_{\mathbb{R}} y f_{Y \mid X}(y \mid x) d y
=\frac{\int_{\mathbb{R}} y f_{X, Y}(x, y) d y}{f_{X}(x)},
 
$$
where $f_{Y \mid X}, f_{X, Y}, f_{X}$ denote the conditional, joint and marginal densities. 
We can estimate those with a kernel $K$ by
$$
\hat{f}_{X}(x)=\frac{\sum_{i=1}^{n} K\left(\frac{x-x_{i}}{h}\right)}{n h}, \hat{f}_{X, Y}(x, y)=\frac{\sum_{i=1}^{n} K\left(\frac{x-x_{i}}{h}\right) K\left(\frac{y-Y_{i}}{h}\right)}{n h^{2}}
$$
Approximating the integral by a Riemann sum this yields the Nadaraya-Watson kernel estimator:
$$\hat{m}(x)=\frac{\sum_{i=1}^{n} K\left(\left(x-x_{i}\right) / h\right) Y_{i}}{\sum_{i=1}^{n} K\left(\left(x-x_{i}\right) / h\right)}$$

**Pros**:
- can be assigned degrees of freedom (trace of the hat-matrix)
- estimation of the noise variance $\hat \sigma_\varepsilon^2$ (XXX c.f. CompStat 3.2.2)

**Cons**:  
- choice of kernel
- if the $x \mapsto K(x)$ is not continuous, $\hat m $ isn't either
- choice of bandwidth, especially if $x_i$ are not equidistant. 

**Examples:**
Normal, Box  
For local bandwidth selection see Brockmann et al. (1993) XXX

### running mean



## Polynomials and Splines
---
### Savitzky–Golay filter
For this section we refer to (Schafer, “What Is a Savitzky-Golay Filter?”). This technique is used in signal processing and can be used to filter out high frequencies . But it can also used for smoothing by filtering high frequency noise while keeping the low frequency signal.  
First we choose a window size $m$. Then, for each point $j \in \{m, m+1, \dots, n-m\}$ we fit a polynomial of degree $k$ by:
$$\hat y_j=\min_{p\in P_k}\sum_{i=-m}^{m}(p (x_{j+i})-y_{i+j})^{2},$$ 
were $P_k$ denotes the Polynomials of degree $k$ over $\mathbb R$.

For equidistant points this can efficiently be calculated by 
$$
\hat y_{j}=\sum_{i=-m}^{m} c_{i} y_{j+i},
$$
where the $c_i$ are only dependent on the $m$ and $k$ and are tabulated in (XXXcitationXXX).

**Pros**
- popular technique in signal processing
- efficient calculation for equidistant points

**Cons**
- no natural way of how to estimate points which are not in the data.

### Polynomial interpolation 

### Polynomial approximation
 
### loess

### Cubic Smoothing Splines
We interpolate with a function in $C^2$ (space of three time continuous differentiable functions) which is defined piecewise by cubic polynomials.
**Pros**
### Regression splines (B-splines)
XXXciteXXX Wood (2017)
use a basis of the spline space (eg B-splines or j-th cardinal basis) and fit the splines of degree k to approximate the data.  

#### B-splines
from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
$$
S(x)=\sum_{j=0}^{n-1} c_{j} B_{j, k ; t}(x)
$$
$$
\begin{array}{r}
B_{i, 0}(x)=1, \text { if } t_{i} \leq x<t_{i+1}, \text { otherwise } 0 \\
B_{i, k}(x)=\frac{x-t_{i}}{t_{i+k}-t_{i}} B_{i, k-1}(x)+\frac{t_{i+k+1}-x}{t_{i+k+1}-t_{i+1}} B_{i+1, k-1}(x)
\end{array}
$$

**Smoothing:**  
We can relax the constrain that we have to perfectly interpolate. Thus we use the minimum number of knots\footnote{SciPy uses FITPACK and DFITPACK, the documentation suggests that smoothness is achieved by reducing the number knots used} such that:
$\sum_{i=1}^n(w (y_i - \hat y_i))^2 \leq s$

**Pros**
- can be assigned degrees of freedom
- extendable to "smooth" version
- performs also well if points are not equidistant

**Cons**
- smoothing process does not translate well to a interpretation (unlike smoothing splines)
- choice of smoothing parameter $s$

### Natural Smoothing Splines
Let $\mathcal F$ be the Sobolev space (the space of functions of which the second derivative is integrable). Then the unique\footnote{Strictly speaking it is only unique for $\lambda > 0$} minimizer 
$$\hat m :=\argmin_{f \in \mathcal F} \sum_{i=1}^{n}\left(Y_{i}-{f}\left(x_{i}\right)\right)^{2}+\lambda \int {f}^{\prime \prime}(x)^{2} d x$$
is a natural\footnote{It is called natural since it is affine outside the data range ($\forall x\notin [x_1, x_n]:\hat m''(x) = 0$)} cubic spline.

**Pros:**  
- can be assigned degrees of freedom (trace of the hat-matrix)
- efficient estimation (closed form solution)
- intuitive penalty (we don't want the function to be too ``wobbly'' --- change slopes)
- performs also well if points are not equidistant
- fixes the Runge's phenomenon (fluctuation of high degree polynomial interpolation)

**Cons:**
- choose $\lambda$

### Penalized Regression Splines
Intuition: similar as Natural Smoothing Splines, but we choose knots

## Kriging
---
\cite{diggleGaussianModelsGeostatistical2007}

#### Idea / Justification /
Kriging was developed in geostatistics to deal with autocorrelation of the response variable at nearby points. By applying the notion that two spectral indices which are (timewise) close should also take similar values we justify the application of Kriging.

#### Definitions and Assumptions
In the end we would like to fit a smooth Gaussian process to the data. 
**Gaussian Process** $\{S(t) : t\in \mathbb R\} $ is a stochastic process if $(S(t_1),\dots,S(t_k))$ has a multivariate Gaussian distribution for every collection of times ${t_1, \dots , t_k}$.   
$S$ can be fully characterized by the mean $\mu(t):=E[S(t)]$ and its covariance function $\gamma\left(t, t^{\prime}\right)=\operatorname{Cov}\left(S(t), S\left(t^{\prime}\right)\right)$

Assumption:  
We will assume the Gaussian process to be stationary. That is for $\mu(t)$ to be constant in $t$ and $\gamma(t,t')$ to depend only on $h=t-t'$. Thus, we will write in the following only $\gamma(h)$.

Note that the process is also isotropic (i.e. $\gamma(h)=\gamma(\|h\|$) since we are in a one-dimensional setting and the covariance is symmetric. 

We also define the variogram of a Gaussian process as 
$$V(h):=V\left(t, t+h\right):=\frac{1}{2} \operatorname{Var}\left(S(t)-S(t+h)\right)\\ %align XXX
=(\gamma(0))^2(1-\operatorname{corr}(S(t),S(t+h)))$$





### Ordinary Kriging
### Universal Kriging
