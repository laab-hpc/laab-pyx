# Report | LAAB-Python | CPU-1x 

| Framework | jax/0.4.25-gfbf-2023a | 
|---|---|
| **System** | kebnekaise:skylake |
| **CPU** | Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz | 
| **LAAB_N** | 3000 |
| **LAAB_REP** | 10 |
| **OMP_NUM_THREADS** | 1 |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Matrix multiplications

The execution times of matrix multiplications invoked through the high-level APIs of the frameworks are compared against those of an optimised reference implementation.

a) **GEMM**:

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | slowdown | loss | result@0.05 | 
|----|------|------------|--|---|--|
|$A^TB$|`transpose(A)@B`| 0.571 | 0.734 | 0.037| :white_check_mark: |
|$"$|`jax.numpy.matmul(transpose(A),B)` | 0.571 |  0.737 | 0.037 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **6.796**| **19.657** | | |
|**Ref (+)** |`sgemm`| **0.329**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.579 | 2.383 | 2.579 | :x: |
|$"$|`jax.numpy.matmul(L,B)`| 0.575 |  2.36 | 2.554  | :x: |
|**Ref (-)** |`sgemm`| **0.329**| **0.924** | | |
|**Ref (+)** |`trmm`| **0.171**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@transpose(A)`| 0.389 | 1.159 | 1.4 | :x: |
|$"$|`jax.numpy.matmul(A,transpose(A))`| 0.39 | 1.166 | 1.409  | :x: |
|**Ref (-)** |`sgemm`| **0.329**| **0.828** | | |
|**Ref (+)** |`syrk`| **0.18**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.57 | 56.092 |1.757 | :x: |
|$"$|`jax.numpy.matmul(T,B)`| 0.575 | 56.587 | 1.772  | :x: | 
|**Ref (-)** |`sgemm`| **0.329**| **31.926** | | |
|**Ref (+)** |`csr(T)@B`| **0.01**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 1.405 | 1.016 | 1.016| :x: |
|**Ref (-)** |`no cse`| **1.394**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.697**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.981 | 0.423 | 0.846 | :x: |
|**Ref (-)** |`no cse`| **2.088**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.392**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.64 | 0.183 | 0.365 | :x: |
|**Ref (-)** |`no cse`| **2.08**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.387**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.642 | 17.473 | 1.309 | :x: |
|**Ref (-)** |`with cse`| **1.276**| **13.35** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.089**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.69 | 14.991 | 0.998 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), H, x])`| 0.044 | 0.0 | 0.0 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.692**| **15.024** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.043**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.014 | 0.025 | 0.001 | :white_check_mark: |  
|$"$|`linalg.multi_dot([transpose(y), transpose(H), H])`| 0.013 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.692**| **50.793** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.013**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.643 | 8.32 | 1.002 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), y, transpose(x), H])`| 0.069 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.642**| **8.306** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.069**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.152 | 0.918 | 0.992 | :x: |
|**Ref (-)** |`no rewrite`| **1.156**| **0.925** | | |
|**Ref (+)**|`A@(B+C)`|**0.601**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.699 | 13.976 | 1.006 | :x: |
|**Ref (-)** |`no rewrite`| **0.695**| **13.899** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.047**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_3$|`transpose(B)@A@transpose(A)@B`| 1.639 | 0.181 | 0.985 | :x: |
|**Ref (-)** |`no rewrite`| **1.643**| **0.184** | | |
|**Ref (+)**|`S = transpose(A)@B; transpose(S)@S`|**1.388**| - | | |



## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.606 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.822**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.607**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.002 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.026**| **12.19** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.573 | 285.255 | 1.0 | :x: |
|**Ref (-)** |`no code motion`| **0.573**| **285.27** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.780 

### Score: 6 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />