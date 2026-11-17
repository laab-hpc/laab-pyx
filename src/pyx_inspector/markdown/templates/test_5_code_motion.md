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

## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
||`{{ config.cm_loops.tests.operator }}`| {{ times.cm_loops.tests.operator }} | {{ slow_down.cm_loops.operator }} | {{ losses.cm_loops.operator }} | {{ cutoff_results.cm_loops.operator }} |
|**Ref (-)** |`{{ config.cm_loops.ref_negative }}`| **{{ times.cm_loops.ref_negative }}**| **{{ slow_down.cm_loops.ref_negative }}** | | |
|**Ref (+)**|`{{ config.cm_loops.ref_positive }}`|**{{ times.cm_loops.ref_positive }}**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
||`{{ config.cm_partial_op_sum.tests.operator }}`| {{ times.cm_partial_op_sum.tests.operator }} | {{ slow_down.cm_partial_op_sum.operator }} | {{ losses.cm_partial_op_sum.operator }} | {{ cutoff_results.cm_partial_op_sum.operator }} |
|**Ref (-)** |`{{ config.cm_partial_op_sum.ref_negative }}`| **{{ times.cm_partial_op_sum.ref_negative }}**| **{{ slow_down.cm_partial_op_sum.ref_negative }}** | | |
|**Ref (+)**|`{{ config.cm_partial_op_sum.ref_positive }}`|**{{ times.cm_partial_op_sum.ref_positive }}**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
||`{{ config.cm_partial_op_prod.tests.operator }}`| {{ times.cm_partial_op_prod.tests.operator }} | {{ slow_down.cm_partial_op_prod.operator }} | {{ losses.cm_partial_op_prod.operator }} | {{ cutoff_results.cm_partial_op_prod.operator }} |
|**Ref (-)** |`{{ config.cm_partial_op_prod.ref_negative }}`| **{{ times.cm_partial_op_prod.ref_negative }}**| **{{ slow_down.cm_partial_op_prod.ref_negative }}** | | |
|**Ref (+)**|`{{ config.cm_partial_op_prod.ref_positive }}`|**{{ times.cm_partial_op_prod.ref_positive }}**| - | | |

## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
