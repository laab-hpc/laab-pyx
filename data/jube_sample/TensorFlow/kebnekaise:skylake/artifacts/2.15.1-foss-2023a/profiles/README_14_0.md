# Report | LAAB-Python | CPU-14x 

| Framework | TensorFlow/2.15.1-foss-2023a | 
|---|---|
| **System** | kebnekaise:skylake |
| **CPU** | Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz | 
| **LAAB_N** | 3000 |
| **LAAB_REP** | 10 |
| **OMP_NUM_THREADS** | 14 |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Matrix multiplications

The execution times of matrix multiplications invoked through the high-level APIs of the frameworks are compared against those of an optimised reference implementation.

a) **GEMM**:

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | slowdown | loss | result@0.05 | 
|----|------|------------|--|---|--|
|$A^TB$|`transpose(A)@B`| 0.04 | 0.078 | 0.006| :white_check_mark: |
|$"$|`linalg.matmul(transpose(A),B)` | 0.04 |  0.079 | 0.006 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **0.487**| **12.154** | | |
|**Ref (+)** |`sgemm`| **0.037**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.041 | 0.217 | 0.0 | :white_check_mark: |
|$"$|`linalg.matmul(L,B)`| 0.04 |  0.185 | 0.0  | :white_check_mark: |
|**Ref (-)** |`sgemm`| **0.037**| **0.0** | | |
|**Ref (+)** |`trmm`| **0.034**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@transpose(A)`| 0.042 | 0.0 | 0.0 | :white_check_mark: |
|$"$|`linalg.matmul(A,transpose(A))`| 0.042 | 0.0 | 0.0  | :white_check_mark: |
|**Ref (-)** |`sgemm`| **0.037**| **0.0** | | |
|**Ref (+)** |`syrk`| **0.152**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.04 | 19.215 |1.098 | :x: |
|$"$|`linalg.matmul(T,B)`| 0.04 | 19.23 | 1.099  | :x: | 
|$"$|`linalg.tridiagonal_matmul(T,B)`| 0.04 | 18.9 | 1.08  | :x: | 
|**Ref (-)** |`sgemm`| **0.037**| **17.5** | | |
|**Ref (+)** |`csr(T)@B`| **0.002**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 0.042 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **0.083**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.041**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 0.081 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no cse`| **0.121**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **0.081**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 0.125 | 0.507 | 1.013 | :x: |
|**Ref (-)** |`no cse`| **0.125**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **0.083**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 0.125 | 34.531 | 1.533 | :x: |
|**Ref (-)** |`with cse`| **0.083**| **22.526** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.004**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.042 | 20.13 | 1.009 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.042**| **19.955** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.002**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.002 | 0.115 | 0.004 | :white_check_mark: |  
|**Ref (-)** |`eval. right to left`| **0.064**| **31.23** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.002**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.05 | 3.404 | 0.976 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.051**| **3.49** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.011**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 0.078 | 0.718 | 1.051 | :x: |
|**Ref (-)** |`no rewrite`| **0.077**| **0.683** | | |
|**Ref (+)**|`A@(B+C)`|**0.045**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.043 | 17.554 | 0.995 | :x: |
|**Ref (-)** |`no rewrite`| **0.043**| **17.649** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.002**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_3$|`transpose(B)@A@transpose(A)@B`| 0.123 | 0.5 | 1.01 | :x: |
|**Ref (-)** |`no rewrite`| **0.123**| **0.495** | | |
|**Ref (+)**|`S = transpose(A)@B; transpose(S)@S`|**0.082**| - | | |



## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.051 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.152**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.051**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.01 | 4.205 | 3.805 | :x: |
|**Ref (-)** |`no code motion`| **0.004**| **1.105** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.043 | 20.72 | 1.182 | :x: |
|**Ref (-)** |`no code motion`| **0.037**| **17.535** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.804 

### Score: 7 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />