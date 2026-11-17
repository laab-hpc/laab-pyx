# CB: {{ cb_id }} · NCPUs: {{ exp_config.omp_num_threads }}

**LAAB-PyX report — {{ exp_config.name }}**

| Framework | {{ eb_name }} | 
|---|---|
| **Test** | {{ test_id }} |
| **CB ID** | {{ cb_id }} |
| **Timestamp** | {{ exp_config.datetime }} |
| **System** | {{ system }} |
| **CPU** | {{ cpu_model }} | 
| **LAAB_N** | {{ exp_config.laab_n }} |
| **LAAB_REP** | {{ exp_config.laab_rep }} |
| **OMP_NUM_THREADS** | {{ exp_config.omp_num_threads }} |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->

## Test 1: Matrix multiplications

The execution times of matrix multiplications invoked through the high-level APIs of the frameworks are compared against those of an optimised reference implementation.

a) **GEMM**:

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | slowdown | loss | result@{{ cutoff }} | 
|----|------|------------|--|---|--|
|$A^TB$|`{{ config.mm_sgemm.tests.operator }}`| {{ times.mm_sgemm.tests.operator }} | {{ slow_down.mm_sgemm.operator }} | {{ losses.mm_sgemm.operator }}| {{ cutoff_results.mm_sgemm.operator }} |
|$"$|`{{ config.mm_sgemm.tests.linalg_matmul }}` | {{ times.mm_sgemm.tests.linalg_matmul }} |  {{ slow_down.mm_sgemm.linalg_matmul }} | {{ losses.mm_sgemm.linalg_matmul }} | {{ cutoff_results.mm_sgemm.linalg_matmul }} |
|**Ref (-)** |`{{ config.mm_sgemm.ref_negative }}`| **{{ times.mm_sgemm.ref_negative }}**| **{{ slow_down.mm_sgemm.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_sgemm.ref_positive }}`| **{{ times.mm_sgemm.ref_positive }}**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|--|
|$LB$|`{{ config.mm_trmm.tests.operator }}`| {{ times.mm_trmm.tests.operator }} | {{ slow_down.mm_trmm.operator }} | {{ losses.mm_trmm.operator }} | {{ cutoff_results.mm_trmm.operator }} |
|$"$|`{{ config.mm_trmm.tests.linalg_matmul }}`| {{ times.mm_trmm.tests.linalg_matmul }} |  {{ slow_down.mm_trmm.linalg_matmul }} | {{ losses.mm_trmm.linalg_matmul }}  | {{ cutoff_results.mm_trmm.linalg_matmul }} |
|**Ref (-)** |`{{ config.mm_trmm.ref_negative }}`| **{{ times.mm_trmm.ref_negative }}**| **{{ slow_down.mm_trmm.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_trmm.ref_positive }}`| **{{ times.mm_trmm.ref_positive }}**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|--|
|$AA^{T}$|`{{ config.mm_syrk.tests.operator }}`| {{ times.mm_syrk.tests.operator }} | {{ slow_down.mm_syrk.operator }} | {{ losses.mm_syrk.operator }} | {{ cutoff_results.mm_syrk.operator }} |
|$"$|`{{ config.mm_syrk.tests.linalg_matmul }}`| {{ times.mm_syrk.tests.linalg_matmul }} | {{ slow_down.mm_syrk.linalg_matmul }} | {{ losses.mm_syrk.linalg_matmul }}  | {{ cutoff_results.mm_syrk.linalg_matmul }} |
|**Ref (-)** |`{{ config.mm_syrk.ref_negative }}`| **{{ times.mm_syrk.ref_negative }}**| **{{ slow_down.mm_syrk.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_syrk.ref_positive }}`| **{{ times.mm_syrk.ref_positive }}**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|--|
|$TB$|`{{ config.mm_tridiag.tests.operator }}`| {{ times.mm_tridiag.tests.operator }} | {{ slow_down.mm_tridiag.operator }} |{{ losses.mm_tridiag.operator }} | {{ cutoff_results.mm_tridiag.operator }} |
|$"$|`{{ config.mm_tridiag.tests.linalg_matmul }}`| {{ times.mm_tridiag.tests.linalg_matmul }} | {{ slow_down.mm_tridiag.linalg_matmul }} | {{ losses.mm_tridiag.linalg_matmul }}  | {{ cutoff_results.mm_tridiag.linalg_matmul }} | {% if config.mm_tridiag.tests.get('linalg_tridiagonal_matmul') %}
|$"$|`{{ config.mm_tridiag.tests.get('linalg_tridiagonal_matmul') }}`| {{ times.mm_tridiag.tests.linalg_tridiagonal_matmul }} | {{ slow_down.mm_tridiag.linalg_tridiagonal_matmul }} | {{ losses.mm_tridiag.linalg_tridiagonal_matmul }}  | {{ cutoff_results.mm_tridiag.linalg_tridiagonal_matmul }} | {% endif %}
|**Ref (-)** |`{{ config.mm_tridiag.ref_negative }}`| **{{ times.mm_tridiag.ref_negative }}**| **{{ slow_down.mm_tridiag.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_tridiag.ref_positive }}`| **{{ times.mm_tridiag.ref_positive }}**| - | | |

## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
