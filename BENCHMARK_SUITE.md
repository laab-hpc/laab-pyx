# LAAB-PyX Benchmark Suite

LAAB-PyX evaluates the linear-algebra awareness of high-level Python
interfaces. The benchmark asks whether a framework executes a mathematical
expression as efficiently as a specialized or algebraically optimized
reference implementation.

The current artifact evaluates PyTorch, TensorFlow, and JAX.

## Test 1: Matrix multiplications

Matrix-matrix multiplication is one of the most expensive operations in many
scientific applications. This test compares high-level framework expressions
with optimized reference implementations:

- **GEMM:** general dense multiplication compared with BLAS `sgemm`.
- **TRMM:** multiplication with a triangular matrix compared with BLAS
  `trmm`, which requires roughly half the work of a general multiplication.
- **SYRK:** a symmetric rank-k update compared with BLAS `syrk`.
- **Tri-diagonal multiplication:** multiplication by a tri-diagonal matrix
  compared with a sparse compressed-row implementation.

## Test 2: Common-subexpression elimination

Repeated subexpressions can sometimes be computed once and reused. For
example, the multiplication in

$$
A^T B + A^T B
$$

appears twice. The benchmark checks repeated expressions in sums and products.
It also includes a case where eliminating the apparent common subexpression
is suboptimal: evaluating a matrix chain from right to left avoids the costly
matrix-matrix product entirely.

## Test 3: Matrix chains

Associativity permits a matrix chain to be parenthesized in different ways.
Those alternatives are mathematically equivalent but can have very different
operation counts. The benchmark covers right-to-left, left-to-right, and
mixed optimal evaluation orders.

## Test 4: Expression rewrites

Algebraic identities can reduce computational cost. These tests cover:

- Two applications of distributivity.
- A transpose-law rewrite that exposes a reusable subexpression.

The optimized references reduce or avoid expensive matrix-matrix products.

## Test 5: Code motion

The code-motion tests determine whether a framework avoids unnecessary work:

- Moving loop-invariant matrix multiplication outside a loop.
- Computing a requested element of a sum without materializing the full sum.
- Computing a requested element of a product without materializing the full
  matrix product.

## Evaluation procedure

Each input expression and its references are measured repeatedly. Let
$\mathbf{t}$ be the input timings and $\mathbf{t}^{+}$ the optimized-reference
timings. The slowdown is

$$
\operatorname{slowdown}(\mathbf{t}, \mathbf{t}^{+}) =
\begin{cases}
\dfrac{\operatorname{median}(\mathbf{t}) -
\operatorname{median}(\mathbf{t}^{+})}
{\operatorname{median}(\mathbf{t}^{+})}, &
Q_{75}(\mathbf{t}^{+}) \le Q_{25}(\mathbf{t}), \\
0, & \text{otherwise}.
\end{cases}
$$

The quantile condition prevents normal timing variability from being reported
as a performance loss.

Because slowdown scales differ between expressions, every input also has a
deliberately suboptimal negative reference with timings $\mathbf{t}^{-}$. The
normalized loss is

$$
\operatorname{loss}(\mathbf{t}, \mathbf{t}^{+}, \mathbf{t}^{-}) =
\frac{\operatorname{slowdown}(\mathbf{t}, \mathbf{t}^{+})}
{\operatorname{slowdown}(\mathbf{t}^{-}, \mathbf{t}^{+})}.
$$

An input passes when its loss is at most the configured cutoff, currently
`0.05`. Each profile reports the mean loss and the number of passing inputs.
