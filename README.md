# Network of noisy resonate and fire neurons Fokker-Planck equation 

Code, relevant figures and  used to illustrate and support:
- the construction of a network of noisy resonate and fire neurons' pde model
- the original associated numerical mass and positivity-preserving solver.

# Description of the model
The model comes from an heuristic mean-field limit of Izhikhevich's resonate and fire model (2001) in a small-kick-size and large number of neurons approximation.
We denote $u_{\rm F}$ the firing voltage threshold and $u_{\rm R}$ the reset voltage.

It has the form of a kinetic Fokker-Planck equation, with $x\leq u_{\rm F}$ being the voltage variable and $v$ the associated velocity, the unknown being a probability density function f(x,v,t). 
The equation reads

```math
\partial_t f + v \partial_x f 
+ \left(-\frac{v}{\tau} - \omega_0^2 x + b\bigl(N(t)+\nu_{\rm ext}\bigr)\right)\partial_v f
- \frac{f}{\tau} 
- a(t)\partial_v^2 f
= N(t)\,\delta_{(u_R,0)},
```

with $a(t)=a_0 + a_1 N(t)$ and $N(t)$ being the activity of the neuron as in the average number of spikes per neuron per second.<br>

The expression of $N$ is:
```math
N(t):=\int_{v>0}vf(u_F,v,t)\,{\rm d}v.
```
# Numerical 
The numerical study is done in a domain 
```math
E:=(x_{\rm min},x_{\rm max})\times (-V,V).
```
with $x_{\rm max}=u_{\rm F}$.

<br>

We implement inflow homogeneous Dirichlet boundary conditions on the side edges, as in
```math
f(u_{\rm F},v<0,t)=0,
```
```math
f(x_{\rm min},v>0,t)=0,
```

<br>


as well as non-influx Robin boundary conditions, which, with, $V$ large enough is, for all $x$ in $(x_{\rm min},x_{\rm max})$:
```math
f(x,V,t)-a(t)\partial_v f(x,V,t)=0,
```
```math
f(x,-V,t)-a(t)\partial_v f(x,-V,t)=0.
```
<br>

The Dirac delta is approximated by a very concentrated maxwellian of mass $1$ centered around point $(u_{\rm R},0)$.

<br>

The used method is an original semi-implicit-like finite differences method with an uniform mesh in $x,v,t$. Mass and positivity preservation properties under mild dynamic conditions can be rigorously proven.

# Contents of the repository

All the directories contain a "Results" sub-directory with outputs of the codes contained in the initial directory.

- 'Solver' contains multiple uses of the pde solver to obtain info on solutions to the pde in different regimes 
- 'Brian 2-single neuron' contains code used to illustrate the behaviour of neurons thanks to the Brian 2 solver [Elife paper](https://elifesciences.org/articles/47314), [Brian2 documentation](https://brian2.readthedocs.io/en/stable/index.html)
- 'Solver-Brian 2-comparison' contains a comparison of the pde model/solver and the Brian 2 solver 
