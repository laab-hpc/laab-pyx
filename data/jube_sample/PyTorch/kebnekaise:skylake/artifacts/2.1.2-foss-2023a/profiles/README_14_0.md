# Report | LAAB-Python | CPU-14x 

| Framework | PyTorch/2.1.2-foss-2023a | 
|---|---|
| **System** | kebnekaise:skylake |
| **CPU** |  | 
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
|$A^TB$|`t(A)@B`| 0.067 | 1.086 | 0.044| :white_check_mark: |
|$"$|`linalg.matmul(t(A),B)` | 0.067 |  1.104 | 0.045 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **0.817**| **24.531** | | |
|**Ref (+)** |`sgemm`| **0.032**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.067 | 2.699 | 3.471 | :x: |
|$"$|`linalg.matmul(L,B)`| 0.067 |  2.694 | 3.464  | :x: |
|**Ref (-)** |`sgemm`| **0.032**| **0.778** | | |
|**Ref (+)** |`trmm`| **0.018**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@t(A)`| 0.066 | 0.0 | 0.0 | :white_check_mark: |
|$"$|`linalg.matmul(A,t(A))`| 0.067 | 0.0 | 0.0  | :white_check_mark: |
|**Ref (-)** |`sgemm`| **0.032**| **0.0** | | |
|**Ref (+)** |`syrk`| **0.148**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.067 | 32.44 |2.163 | :x: |
|$"$|`linalg.matmul(T,B)`| 0.066 | 32.075 | 2.138  | :x: | 
|**Ref (-)** |`sgemm`| **0.032**| **15.0** | | |
|**Ref (+)** |`csr(T)@B`| **0.002**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | 0.072 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **0.144**| **1.0** | | | 
|**Ref (+)**| `2*(t(A)@B)`| **0.072**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| 0.136 | 0.016 | 0.032 | :white_check_mark: |
|**Ref (-)** |`no cse`| **0.2**| **0.5** | | |
|**Ref (+)**| `S=t(A)@B; t(S)@S`| **0.133**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`t(t(A)@B)@t(A)@B`| 0.203 | 0.499 | 0.998 | :x: |
|**Ref (-)** |`no cse`| **0.203**| **0.5** | | |
|**Ref (+)**| `S=t(A)@B; t(S)@S`| **0.135**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`t(A)@B@t(A)@B@y`| 0.202 | 77.104 | 1.494 | :x: |
|**Ref (-)** |`with cse`| **0.136**| **51.618** | | |
|**Ref (+)**| `t(A)@(B@(t(A)@(B@y))`| **0.003**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`t(H)@H@x`| 0.068 | 32.93 | 1.009 | :x: | 
|$"$|`linalg.multi_dot([t(H), H, x])`| 0.001 | 0.0 | 0.0 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.067**| **32.63** | | |
|**Ref (+)**| `t(H)@(H@x)`| **0.002**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| 0.002 | 0.0 | 0.0 | :white_check_mark: |  
|$"$|`linalg.multi_dot([t(y), t(H), H])`| 0.001 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.068**| **32.845** | | |
|**Ref (+)**| `(t(y)@t(H))@H`| **0.002**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| 0.105 | 1.741 | 1.01 | :x: | 
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| 0.038 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.104**| **1.724** | | |
|**Ref (+)**| `(t(H)@y)@(t(x)@H)`| **0.038**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 0.14 | 0.938 | 1.019 | :x: |
|**Ref (-)** |`no rewrite`| **0.138**| **0.92** | | |
|**Ref (+)**|`A@(B+C)`|**0.072**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - t(H)@H)@x`| 0.073 | 35.635 | 1.013 | :x: |
|**Ref (-)** |`no rewrite`| **0.072**| **35.17** | | |
|**Ref (+)**|`A@x - t(H)@(H@x)`|**0.002**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_3$|`t(B)@A@t(A)@B`| 0.202 | 0.511 | 1.021 | :x: |
|**Ref (-)** |`no rewrite`| **0.201**| **0.501** | | |
|**Ref (+)**|`S = t(A)@B; t(S)@S`|**0.134**| - | | |



## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.109 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.327**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.109**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.005 | 1.122 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.003**| **0.0** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.066 | 27.665 | 1.038 | :x: |
|**Ref (-)** |`no code motion`| **0.064**| **26.665** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.721 

### Score: 9 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />