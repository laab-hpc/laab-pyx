# Report | LAAB-Python | CPU-1x 

| Framework | TensorFlow/2.15.1-foss-2023a | 
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
|$A^TB$|`transpose(A)@B`| 0.377 | 0.299 | 0.014| :white_check_mark: |
|$"$|`linalg.matmul(transpose(A),B)` | 0.377 |  0.302 | 0.014 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **6.405**| **21.085** | | |
|**Ref (+)** |`sgemm`| **0.29**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.383 | 1.587 | 1.654 | :x: |
|$"$|`linalg.matmul(L,B)`| 0.385 |  1.602 | 1.67  | :x: |
|**Ref (-)** |`sgemm`| **0.29**| **0.959** | | |
|**Ref (+)** |`trmm`| **0.148**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@transpose(A)`| 0.4 | 1.549 | 1.828 | :x: |
|$"$|`linalg.matmul(A,transpose(A))`| 0.404 | 1.573 | 1.857  | :x: |
|**Ref (-)** |`sgemm`| **0.29**| **0.847** | | |
|**Ref (+)** |`syrk`| **0.157**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.366 | 38.626 |1.272 | :x: |
|$"$|`linalg.matmul(T,B)`| 0.387 | 40.843 | 1.345  | :x: | 
|$"$|`linalg.tridiagonal_matmul(T,B)`| 0.036 | 2.873 | 0.095  | :x: | 
|**Ref (-)** |`sgemm`| **0.29**| **30.365** | | |
|**Ref (+)** |`csr(T)@B`| **0.009**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 0.375 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **0.747**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.373**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 0.744 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no cse`| **1.096**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **0.731**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.211 | 0.552 | 1.104 | :x: |
|**Ref (-)** |`no cse`| **1.171**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **0.781**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.188 | 114.589 | 1.615 | :x: |
|**Ref (-)** |`with cse`| **0.74**| **70.951** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.01**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.38 | 70.492 | 1.0 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.38**| **70.476** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.005**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.006 | 0.115 | 0.001 | :white_check_mark: |  
|**Ref (-)** |`eval. right to left`| **0.389**| **77.523** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.005**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.415 | 3.055 | 1.003 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.414**| **3.047** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.102**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 0.772 | 0.883 | 0.998 | :x: |
|**Ref (-)** |`no rewrite`| **0.773**| **0.885** | | |
|**Ref (+)**|`A@(B+C)`|**0.41**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.405 | 48.935 | 1.017 | :x: |
|**Ref (-)** |`no rewrite`| **0.398**| **48.105** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.008**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_3$|`transpose(B)@A@transpose(A)@B`| 1.198 | 0.536 | 1.079 | :x: |
|**Ref (-)** |`no rewrite`| **1.168**| **0.496** | | |
|**Ref (+)**|`S = transpose(A)@B; transpose(S)@S`|**0.78**| - | | |



## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.481 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.421**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.474**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.03 | 14.17 | 1.154 | :x: |
|**Ref (-)** |`no code motion`| **0.027**| **12.275** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.388 | 193.125 | 1.035 | :x: |
|**Ref (-)** |`no code motion`| **0.375**| **186.62** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.800 

### Score: 5 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />