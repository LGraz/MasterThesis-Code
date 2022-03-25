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

## Polynomial interpolation 

## Polynomial approximation

---
## Splines
### Cubic Smoothing Splines
We interpolate with a function in $C^2$ (space of three time continious differentiable functions) which is defined piecwise by cubic polinomials.
**Pros**
### Regression splines (B-splines)
XXXciteXXX Wood (2017)
use a basis of the spline space (eg B-splines or j-th cardinal basis) and fit the first **k** splines to approximate the data.  


**Pros**

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