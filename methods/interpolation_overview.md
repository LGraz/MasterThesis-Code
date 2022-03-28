# Table - overview
parameter: {h: bandwidth, $\lambda$: penalty-factor, k: #neighbours}



## categorys
tuning parameter  
unbiased  
asympt. unbiased  
deg. of freedom  
continous  
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
Approximating the integral by a rieman sum this yields the Nadaraya-Watson kernel estimator:
$$\hat{m}(x)=\frac{\sum_{i=1}^{n} K\left(\left(x-x_{i}\right) / h\right) Y_{i}}{\sum_{i=1}^{n} K\left(\left(x-x_{i}\right) / h\right)}$$

**Pros**:
- can be assigned degrees of freedom (trace of the hat-matrix)
- estimation of the noise variance $\hat \sigma_\varepsilon^2$ (XXX c.f. CompStat 3.2.2)

**Cons**:  
- choice of kernel
- if the $x \mapsto K(x)$ is not continious, $\hat m $ isn't either
- choice of bandwidth, especially if $x_i$ are not equidistant. 

**Examples:**
Normal, Box  
For local bandwidth selection see Brockmann et al. (1993) XXX

### running mean

### loess

## Savitzky–Golay filter

## Polynomial interpolation 

## Polynomial approximation

---
## Splines
### Cubic Smoothing Splines
We interpolate with a function in $C^2$ (space of three time continious differentiable functions) which is defined piecwise by cubic polinomials.
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
We can relax the constrain that we have to perfectly interpolate. Thus we use the minimum number of knots\footnote{SciPy uses FITPACK and DFITPACK, the documentation suggests that smoothness is achived by reducing the number knots used} such that:
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
$$\hat m :=\argmin_{f \in \mathcal F} \sum_{i=1}^{n}\left\{Y_{i}-{f}\left(x_{i}\right)\right\}^{2}+\lambda \int {f}^{\prime \prime}(x)^{2} d x$$
is a natural\footnote{It is called natural since it is affine outside of the data range ($\forall x\notin [x_1, x_n]:\hat m''(x) = 0$)} cubic spline.

**Pros:**  
- can be assigned degrees of freedom (trace of the hat-matrix)
- efficient estimation (closed form solution)
- intuitive penalty (we don't want the function to be too ``wobbly'' --- change slopes)
- performs also well if points are not equidistant
- fixes the Runge's phenomenon (fluctuation of high degree polinomial interpolation)

**Cons:**
- choose $\lambda$

### Penalized Regression Splines
Intuition: similar as Natural Smoothing Splines but we choose knots