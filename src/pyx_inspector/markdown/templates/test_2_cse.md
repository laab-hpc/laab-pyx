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

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_1$ |`{{ config.cse_addition.tests.operator }}` | {{ times.cse_addition.tests.operator }} | {{ slow_down.cse_addition.operator }} | {{ losses.cse_addition.operator }}| {{ cutoff_results.cse_addition.operator }} |
|**Ref (-)** |`{{ config.cse_addition.ref_negative }}`| **{{ times.cse_addition.ref_negative }}**| **{{ slow_down.cse_addition.ref_negative }}** | | | 
|**Ref (+)**| `{{ config.cse_addition.ref_positive }}`| **{{ times.cse_addition.ref_positive }}**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_2$|`{{ config.cse_matmul_paranthesis.tests.operator }}`| {{ times.cse_matmul_paranthesis.tests.operator }} | {{ slow_down.cse_matmul_paranthesis.operator }} | {{ losses.cse_matmul_paranthesis.operator }} | {{ cutoff_results.cse_matmul_paranthesis.operator }} |
|**Ref (-)** |`{{ config.cse_matmul_paranthesis.ref_negative }}`| **{{ times.cse_matmul_paranthesis.ref_negative }}**| **{{ slow_down.cse_matmul_paranthesis.ref_negative }}** | | |
|**Ref (+)**| `{{ config.cse_matmul_paranthesis.ref_positive }}`| **{{ times.cse_matmul_paranthesis.ref_positive }}**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_3$|`{{ config.cse_matmul_no_paranthesis.tests.operator }}`| {{ times.cse_matmul_no_paranthesis.tests.operator }} | {{ slow_down.cse_matmul_no_paranthesis.operator }} | {{ losses.cse_matmul_no_paranthesis.operator }} | {{ cutoff_results.cse_matmul_no_paranthesis.operator }} |
|**Ref (-)** |`{{ config.cse_matmul_no_paranthesis.ref_negative }}`| **{{ times.cse_matmul_no_paranthesis.ref_negative }}**| **{{ slow_down.cse_matmul_no_paranthesis.ref_negative }}** | | |
|**Ref (+)**| `{{ config.cse_matmul_no_paranthesis.ref_positive }}`| **{{ times.cse_matmul_no_paranthesis.ref_positive }}**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$ and $y \in \mathbb{R}^{ {{ exp_config.laab_n }} }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_4$|`{{ config.cse_suboptimal.tests.operator }}`| {{ times.cse_suboptimal.tests.operator }} | {{ slow_down.cse_suboptimal.operator }} | {{ losses.cse_suboptimal.operator }} | {{ cutoff_results.cse_suboptimal.operator }} |
|**Ref (-)** |`{{ config.cse_suboptimal.ref_negative }}`| **{{ times.cse_suboptimal.ref_negative }}**| **{{ slow_down.cse_suboptimal.ref_negative }}** | | |
|**Ref (+)**| `{{ config.cse_suboptimal.ref_positive }}`| **{{ times.cse_suboptimal.ref_positive }}**| - | | |

## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
