# Report | LAAB-Python | CPU-4x 

| Framework | jax/0.4.25-gfbf-2023a | 
|---|---|
| **System** | kebnekaise:skylake |
| **CPU** | Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz | 
| **LAAB_N** | 3000 |
| **LAAB_REP** | 10 |
| **OMP_NUM_THREADS** | 4 |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Matrix multiplications

The execution times of matrix multiplications invoked through the high-level APIs of the frameworks are compared against those of an optimised reference implementation.

a) **GEMM**:

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | slowdown | loss | result@0.05 | 
|----|------|------------|--|---|--|
|$A^TB$|`transpose(A)@B`| 0.153 | 0.822 | 0.042| :white_check_mark: |
|$"$|`jax.numpy.matmul(transpose(A),B)` | 0.152 |  0.805 | 0.041 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **1.737**| **19.674** | | |
|**Ref (+)** |`sgemm`| **0.084**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.15 | 2.256 | 2.731 | :x: |
|$"$|`jax.numpy.matmul(L,B)`| 0.151 |  2.286 | 2.767  | :x: |
|**Ref (-)** |`sgemm`| **0.084**| **0.826** | | |
|**Ref (+)** |`trmm`| **0.046**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@transpose(A)`| 0.1 | 0.0 | 0.0 | :white_check_mark: |
|$"$|`jax.numpy.matmul(A,transpose(A))`| 0.1 | 0.0 | 0.0  | :white_check_mark: |
|**Ref (-)** |`sgemm`| **0.084**| **0.0** | | |
|**Ref (+)** |`syrk`| **0.177**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.151 | 67.53 |1.82 | :x: |
|$"$|`jax.numpy.matmul(T,B)`| 0.149 | 66.65 | 1.796  | :x: | 
|**Ref (-)** |`sgemm`| **0.084**| **37.113** | | |
|**Ref (+)** |`csr(T)@B`| **0.002**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 0.398 | 1.015 | 1.015| :x: |
|**Ref (-)** |`no cse`| **0.395**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.197**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 0.543 | 0.444 | 0.888 | :x: |
|**Ref (-)** |`no cse`| **0.564**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **0.376**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 0.451 | 0.131 | 0.262 | :x: |
|**Ref (-)** |`no cse`| **0.598**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **0.399**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 0.443 | 4.096 | 1.351 | :x: |
|**Ref (-)** |`with cse`| **0.35**| **3.032** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.087**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.193 | 3.439 | 0.985 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), H, x])`| 0.043 | 0.0 | 0.0 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.195**| **3.49** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.043**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.014 | 0.022 | 0.002 | :white_check_mark: |  
|$"$|`linalg.multi_dot([transpose(y), transpose(H), H])`| 0.013 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.196**| **13.837** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.013**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.204 | 2.522 | 1.019 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), y, transpose(x), H])`| 0.058 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.202**| **2.476** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.058**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 0.299 | 0.829 | 1.002 | :x: |
|**Ref (-)** |`no rewrite`| **0.298**| **0.828** | | |
|**Ref (+)**|`A@(B+C)`|**0.163**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.201 | 3.338 | 0.974 | :x: |
|**Ref (-)** |`no rewrite`| **0.205**| **3.427** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.046**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_3$|`transpose(B)@A@transpose(A)@B`| 0.428 | 0.129 | 0.722 | :x: |
|**Ref (-)** |`no rewrite`| **0.446**| **0.179** | | |
|**Ref (+)**|`S = transpose(A)@B; transpose(S)@S`|**0.379**| - | | |



## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.17 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.506**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.169**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.002 | 0.203 | 0.033 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.014**| **6.065** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.156 | 76.8 | 1.063 | :x: |
|**Ref (-)** |`no code motion`| **0.146**| **72.215** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.699 

### Score: 7 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />