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

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$, $x \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|----|---------|--|--|--|
|$H^THx$|`{{ config.matchain_rtol.tests.operator }}`| {{ times.matchain_rtol.tests.operator }} | {{ slow_down.matchain_rtol.operator }} | {{ losses.matchain_rtol.operator }} | {{ cutoff_results.matchain_rtol.operator }} | {% if config.matchain_rtol.tests.get('linalg_multidot') %}
|$"$|`{{ config.matchain_rtol.tests.get('linalg_multidot') }}`| {{ times.matchain_rtol.tests.linalg_multidot }} | {{ slow_down.matchain_rtol.linalg_multidot }} | {{ losses.matchain_rtol.linalg_multidot }} | {{ cutoff_results.matchain_rtol.linalg_multidot }} |  {% endif %}
|**Ref (-)** |`{{ config.matchain_rtol.ref_negative }}`| **{{ times.matchain_rtol.ref_negative }}**| **{{ slow_down.matchain_rtol.ref_negative }}** | | |
|**Ref (+)**| `{{ config.matchain_rtol.ref_positive }}`| **{{ times.matchain_rtol.ref_positive }}**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$, $y \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@{{ cutoff }} |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`{{ config.matchain_ltor.tests.operator }}`| {{ times.matchain_ltor.tests.operator }} | {{ slow_down.matchain_ltor.operator }} | {{ losses.matchain_ltor.operator }} | {{ cutoff_results.matchain_ltor.operator }} |  {% if config.matchain_ltor.tests.get('linalg_multidot') %}
|$"$|`{{ config.matchain_ltor.tests.get('linalg_multidot') }}`| {{ times.matchain_ltor.tests.linalg_multidot }} | {{ slow_down.matchain_ltor.linalg_multidot }} | {{ losses.matchain_ltor.linalg_multidot }} | {{ cutoff_results.matchain_ltor.linalg_multidot }} | {% endif %}
|**Ref (-)** |`{{ config.matchain_ltor.ref_negative }}`| **{{ times.matchain_ltor.ref_negative }}**| **{{ slow_down.matchain_ltor.ref_negative }}** | | |
|**Ref (+)**| `{{ config.matchain_ltor.ref_positive }}`| **{{ times.matchain_ltor.ref_positive }}**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$ and $x,y \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@{{ cutoff }} |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`{{ config.matchain_mixed.tests.operator }}`| {{ times.matchain_mixed.tests.operator }} | {{ slow_down.matchain_mixed.operator }} | {{ losses.matchain_mixed.operator }} | {{ cutoff_results.matchain_mixed.operator }} | {% if config.matchain_mixed.tests.get('linalg_multidot') %}
|$"$|`{{ config.matchain_mixed.tests.get('linalg_multidot') }}`| {{ times.matchain_mixed.tests.linalg_multidot }} | {{ slow_down.matchain_mixed.linalg_multidot }} | {{ losses.matchain_mixed.linalg_multidot }} | {{ cutoff_results.matchain_mixed.linalg_multidot }} | {% endif %}
|**Ref (-)** |`{{ config.matchain_mixed.ref_negative }}`| **{{ times.matchain_mixed.ref_negative }}**| **{{ slow_down.matchain_mixed.ref_negative }}** | | |
|**Ref (+)**| `{{ config.matchain_mixed.ref_positive }}`| **{{ times.matchain_mixed.ref_positive }}**| - | | |

## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
