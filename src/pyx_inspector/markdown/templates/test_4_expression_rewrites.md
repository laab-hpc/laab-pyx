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

## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$E_1$|`{{ config.am_distributivity1.tests.operator }}`| {{ times.am_distributivity1.tests.operator }} | {{ slow_down.am_distributivity1.operator }} | {{ losses.am_distributivity1.operator }} | {{ cutoff_results.am_distributivity1.operator }} |
|**Ref (-)** |`{{ config.am_distributivity1.ref_negative }}`| **{{ times.am_distributivity1.ref_negative }}**| **{{ slow_down.am_distributivity1.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_distributivity1.ref_positive }}`|**{{ times.am_distributivity1.ref_positive }}**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$E_2$|`{{ config.am_distributivity2.tests.operator }}`| {{ times.am_distributivity2.tests.operator }} | {{ slow_down.am_distributivity2.operator }} | {{ losses.am_distributivity2.operator }} | {{ cutoff_results.am_distributivity2.operator }} |
|**Ref (-)** |`{{ config.am_distributivity2.ref_negative }}`| **{{ times.am_distributivity2.ref_negative }}**| **{{ slow_down.am_distributivity2.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_distributivity2.ref_positive }}`|**{{ times.am_distributivity2.ref_positive }}**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$E_3$|`{{ config.am_transpose.tests.operator }}`| {{ times.am_transpose.tests.operator }} | {{ slow_down.am_transpose.operator }} | {{ losses.am_transpose.operator }} | {{ cutoff_results.am_transpose.operator }} |
|**Ref (-)** |`{{ config.am_transpose.ref_negative }}`| **{{ times.am_transpose.ref_negative }}**| **{{ slow_down.am_transpose.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_transpose.ref_positive }}`|**{{ times.am_transpose.ref_positive }}**| - | | |

## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
