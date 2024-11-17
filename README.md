# An Efficient Quasi-Newton Method with Tensor Product Implementation for Solving Quasi-Linear Elliptic Equations and Systems

This repository contains an efficient quasi-Newton method tailored for solving quasi-linear elliptic equations. The method leverages GPU architectures and tensor product structures to enhance computational efficiency. By approximating the Jacobian matrix using the linear Laplacian operator along with simplified representations of the nonlinear terms, we effectively transform the quasi-linear problem into a form more amenable to parallel computation and GPU acceleration.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Quasi-Newton Approach](#quasi-newton-approach)
  - [Exploiting Tensor and Kronecker Products](#exploiting-tensor-and-kronecker-products)
- [Implementation Details](#implementation-details)
- [References](#references)

## Introduction

Many discretization schemes for partial differential equations lead to linear systems characterized by sparse matrices, where nonlinear terms can often be approximated by diagonal or block-diagonal matrices. Discretizing the equations using appropriate numerical methods leads to a nonlinear system:

$$
    A_h U_h + N_h(U_h) = 0,
$$

where:

- $A_h$ is the discretized Laplacian operator or linear operator.
- $N_h(U_h)$ represents the discretized nonlinear terms.
- $U_h$ is the numerical solution.

Traditional Newton's methods require the inversion of the Jacobian matrix $J_h = A_h + N_u(U_h)$ at each iteration, which becomes computationally demanding as the matrix size and complexity increase. Our quasi-Newton approach simplifies the inversion process, enabling more efficient computation while ensuring convergence under proper assumptions.

## Methodology

### Quasi-Newton Approach

Our method approximates the Jacobian inverse by incorporating a scaled identity matrix as a proxy for the nonlinear terms:

$$
    [A_h+\beta I]^{-1} \approx [A_h+N_u]^{-1},
$$

where:

- $\beta$ is a constant scaling factor.
- $I$ is the identity matrix.
- $N_u$ is the Jacobian of the nonlinear term.

The primary computational advantage is that we compute the diagonalization of $A_h$ once and reuse it throughout the iterations. Expressing $A_h$ in its diagonalized form:

$$
    A_h = T \Lambda T^{-1},
$$

where:

- $T$ contains the eigenvectors.
- $\Lambda$ is the diagonal matrix of eigenvalues.

This leads to:

$$
    A_h + \beta I = T (\Lambda + \beta I ) T^{-1}.
$$

Inverting $A_h + \beta I$ then involves only the inversion of the diagonal matrix $\Lambda + \beta I$ , simplifying computations significantly.

### Exploiting Tensor and Kronecker Products

Our approach further exploits the structure of tensor and Kronecker products to enhance efficiency. For matrices arising from discretizations on tensor-product grids, $A_h$ can often be expressed as:

$$
    A_h = I_y \otimes A_x + A_y \otimes I_x,
$$

where:

- $A_x$ and $A_y$ are one-dimensional discretization matrices.
- $I_x$ and $I_y$ are identity matrices.
- $\otimes$ denotes the Kronecker product.

By diagonalizing $A_x$ and $A_y$:

$$
    A_x = T_x \Lambda_x T_x^{-1}, \quad A_y = T_y \Lambda_y T_y^{-1},
$$

we can express $A_h + \beta I$ as:

$$
    A_h + \beta I = (T_y \otimes T_x) \left( I_y \otimes (\Lambda_x + \beta I) + (\Lambda_y + \beta I) \otimes I_x \right) (T_y^{-1} \otimes T_x^{-1}).
$$

Since $\Lambda_x$ and $\Lambda_y$ are diagonal, the inversion reduces to element-wise operations, which are highly efficient and suitable for GPU acceleration.

## Implementation Details

Our implementation builds upon a minimalist MATLAB code originally developed by [Liu et al. (2023)](https://www.math.purdue.edu/~zhan1966/research/code/PoissonGPU.html). We have adapted this code to handle the nonlinear terms by employing our quasi-Newton method, which approximates the nonlinear term with a scaled identity matrix.

While the Spectral Element Method (SEM) is emphasized due to its greater accuracy and complexity, the code can be adapted for the Finite Difference Method (FDM) with straightforward modifications.


