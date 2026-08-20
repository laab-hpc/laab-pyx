import pickle
from pathlib import Path

from flask import Blueprint, render_template, request

from .bootstrap import PROFILES

name = Path(__file__).parent.name

bp = Blueprint(
    name,
    __name__,
    template_folder="templates/",
    static_folder="static/",
    static_url_path=f"/static/{name}/",
)

COMPARE_GROUPS = [
    (
        "Test 1: Matrix multiplications",
        [
            ("mm_sgemm", "gemm"),
            ("mm_trmm", "trmm"),
            ("mm_syrk", "syrk"),
            ("mm_tridiag", "Tridiagonal"),
        ],
    ),
    (
        "Test 2: CSE",
        [
            ("cse_addition", "Repeated in summation"),
            ("cse_matmul_paranthesis", "Repeated in product (parenthesis)"),
            ("cse_matmul_no_paranthesis", "Repeated in product (no-parenthesis)"),
            ("cse_suboptimal", "Suboptimal CSE"),
        ],
    ),
    (
        "Test 3: Matrix chains",
        [
            ("matchain_rtol", "Right-to-left"),
            ("matchain_ltor", "Left-to-right"),
            ("matchain_mixed", "Mixed"),
        ],
    ),
    (
        "Test 4: Expression rewrites",
        [
            ("am_distributivity1", "Distributivity 1"),
            ("am_distributivity2", "Distributivity 2"),
            ("am_transpose", "Transpose law"),
        ],
    ),
    (
        "Test 5: Code motion",
        [
            ("cm_loops", "Loop-invariant code motion"),
            ("cm_partial_op_sum", "Partial op access in sum"),
            ("cm_partial_op_prod", "Partial op access in product"),
        ],
    ),
]

EXPERIMENT_METADATA = {
    "mm_sgemm": {
        "title": "Test 1a: GEMM",
        "expr_header": "",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The time taken for general matrix multiplication A<sup>T</sup>B is compared for equivalence against the reference <code>sgemm</code> routine invoked via OpenBLAS from C.",
        "labels": {"operator": "A<sup>T</sup>B", "linalg_matmul": '"'},
        "ref_negative_label": "sgemv for each row",
    },
    "mm_trmm": {
        "title": "Test 1b: TRMM",
        "expr_header": "Expr",
        "operands": "L, B in R<sup>n x n</sup>",
        "description": "The input expression is LB, where L is lower triangular. The reference implementation utilises the BLAS kernel <code>trmm</code>, which computes the matrix product with half the number of FLOPs than that required by <code>gemm</code>.",
        "labels": {"operator": "LB", "linalg_matmul": '"'},
        "ref_negative_label": "sgemm",
    },
    "mm_syrk": {
        "title": "Test 1c: SYRK",
        "expr_header": "Expr",
        "operands": "A in R<sup>n x n</sup>",
        "description": 'The input expression is AA<sup>T</sup>. The reference implementation utilises the BLAS routine <code>syrk</code> ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than <code>gemm</code>.',
        "labels": {"operator": "AA<sup>T</sup>", "linalg_matmul": '"'},
        "ref_negative_label": "sgemm",
    },
    "mm_tridiag": {
        "title": "Test 1d: Tri-diagonal",
        "expr_header": "Expr",
        "operands": "T, B in R<sup>n x n</sup>",
        "description": "The input expression is TB, where T is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for T, implemented in C.",
        "labels": {
            "operator": "TB",
            "linalg_matmul": '"',
            "linalg_tridiagonal_matmul": '"',
        },
        "ref_negative_label": "sgemm",
    },
    "cse_addition": {
        "title": "Test 2a: Repeated in summation",
        "expr_header": "Expr",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is E<sub>1</sub> = A<sup>T</sup>B + A<sup>T</sup>B. The subexpression A<sup>T</sup>B appears twice. The execution time to evaluate E<sub>1</sub> is compared for equivalence against a reference implementation that computes A<sup>T</sup>B just once.",
        "labels": {"operator": "E<sub>1</sub>"},
        "ref_negative_label": "no cse",
    },
    "cse_matmul_paranthesis": {
        "title": "Test 2b: Repeated in multiplication (parenthesis)",
        "expr_header": "Expr",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is E<sub>2</sub> = (A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B). The reference implementation avoids the redundant computation of the common subexpression.",
        "labels": {"operator": "E<sub>2</sub>"},
        "ref_negative_label": "no cse",
    },
    "cse_matmul_no_paranthesis": {
        "title": "Test 2c: Repeated in multiplication (no parenthesis)",
        "expr_header": "Expr",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is E<sub>3</sub> = (A<sup>T</sup>B)<sup>T</sup>A<sup>T</sup>B. The reference implementation avoids the redundant computation of the common subexpression.",
        "labels": {"operator": "E<sub>3</sub>"},
        "ref_negative_label": "no cse",
    },
    "cse_suboptimal": {
        "title": "Test 2d: Sub-optimal CSE",
        "expr_header": "Expr",
        "operands": "A, B in R<sup>n x n</sup> and y in R<sup>n</sup>",
        "description": "The input expression is E<sub>4</sub> = A<sup>T</sup>BA<sup>T</sup>By. The reference implementation evaluates E<sub>4</sub> from right-to-left without CSE.",
        "labels": {"operator": "E<sub>4</sub>"},
        "ref_negative_label": "with cse",
    },
    "matchain_rtol": {
        "title": "Test 3a: Right to left",
        "expr_header": "Expr",
        "operands": "H in R<sup>n x n</sup>, x in R<sup>n</sup>",
        "description": "The input matrix chain is H<sup>T</sup>Hx. The reference implementation, evaluating from right to left, i.e. H<sup>T</sup>(Hx), avoids the expensive O(n<sup>3</sup>) matrix product and has complexity O(n<sup>2</sup>).",
        "labels": {"operator": "H<sup>T</sup>Hx", "linalg_multidot": '"'},
        "ref_negative_label": "eval. left to right",
    },
    "matchain_ltor": {
        "title": "Test 3b: Left to right",
        "expr_header": "Expr",
        "operands": "H in R<sup>n x n</sup>, y in R<sup>n</sup>",
        "description": "The input matrix chain is y<sup>T</sup>H<sup>T</sup>H. The reference implementation, evaluating from left to right, i.e. (y<sup>T</sup>H<sup>T</sup>)H, avoids the expensive O(n<sup>3</sup>) matrix product and has complexity O(n<sup>2</sup>).",
        "labels": {"operator": "y<sup>T</sup>H<sup>T</sup>H", "linalg_multidot": '"'},
        "ref_negative_label": "eval. right to left",
    },
    "matchain_mixed": {
        "title": "Test 3c: Mixed",
        "expr_header": "Expr",
        "operands": "H in R<sup>n x n</sup> and x, y in R<sup>n</sup>",
        "description": "The input matrix chain is H<sup>T</sup>yx<sup>T</sup>H. Here, neither left-to-right nor right-to-left evaluation avoids the expensive O(n<sup>3</sup>) operation; instead, the evaluation (H<sup>T</sup>y)(x<sup>T</sup>H) turns out to be the optimum with O(n<sup>2</sup>) complexity.",
        "labels": {"operator": "H<sup>T</sup>yx<sup>T</sup>H", "linalg_multidot": '"'},
        "ref_negative_label": "eval. left to right",
    },
    "am_distributivity1": {
        "title": "Test 4a: Distributivity 1",
        "expr_header": "Expr",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is E<sub>1</sub> = AB + AC. This expression requires two O(n<sup>3</sup>) matrix multiplications. E<sub>1</sub> can be rewritten using the distributive law as A(B + C), reducing the number of O(n<sup>3</sup>) matrix multiplications to one.",
        "labels": {"operator": "E<sub>1</sub>"},
        "ref_negative_label": "no rewrite",
    },
    "am_distributivity2": {
        "title": "Test 4b: Distributivity 2",
        "expr_header": "Expr",
        "operands": "A, H in R<sup>n x n</sup>",
        "description": "The input expression is E<sub>2</sub> = (A - H<sup>T</sup>H)x, which involves one O(n<sup>3</sup>) matrix multiplication. This expression can be rewritten as Ax - H<sup>T</sup>(Hx), thereby avoiding the O(n<sup>3</sup>) matrix multiplication.",
        "labels": {"operator": "E<sub>2</sub>"},
        "ref_negative_label": "no rewrite",
    },
    "am_transpose": {
        "title": "Test 4c: Transpose law",
        "expr_header": "Expr",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is E<sub>3</sub> = B<sup>T</sup>AA<sup>T</sup>B. This expression can be rewritten as (A<sup>T</sup>B)<sup>T</sup>(A<sup>T</sup>B) by applying the transpose law and the sub-expression A<sup>T</sup>B can be computed just once.",
        "labels": {"operator": "E<sub>3</sub>"},
        "ref_negative_label": "no rewrite",
    },
    "cm_loops": {
        "title": "Test 5a: Loop-invariant code motion",
        "expr_header": "",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is AB computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.",
        "labels": {"operator": ""},
        "ref_negative_label": "no code motion",
    },
    "cm_partial_op_sum": {
        "title": "Test 5b: Partial operand access in sum",
        "expr_header": "",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is (A + B)[2,2], which requires only a single element of both matrices. The reference implementation avoids the explicit addition of the full matrices.",
        "labels": {"operator": ""},
        "ref_negative_label": "no code motion",
    },
    "cm_partial_op_prod": {
        "title": "Test 5c: Partial operand access in product",
        "expr_header": "",
        "operands": "A, B in R<sup>n x n</sup>",
        "description": "The input expression is (AB)[2,2], which requires only a single element of both matrices. The reference implementation avoids the explicit multiplication of the full matrices.",
        "labels": {"operator": ""},
        "ref_negative_label": "no code motion",
    },
}


def _humanize_result(result):
    if result == ":white_check_mark:":
        return "pass"
    if result == ":x:":
        return "fail"
    return result


def _extract_cb_id(profile_path: str) -> str:
    parts = Path(profile_path).stem.split("_")
    if len(parts) < 3:
        return ""
    return parts[-1]


def _result_symbol(result: str) -> str:
    if result == ":white_check_mark:":
        return "✓"
    if result == ":x:":
        return "✗"
    return ""


def _load_profile(profile_path: str):
    profile_file = PROFILES.profiles_root / profile_path
    with profile_file.open("rb") as handle:
        return pickle.load(handle)


def _build_common_metadata(profiles):
    candidates = [
        ("Library", [p.get("eb_name", "").split("/", 1)[0] if p.get("eb_name", "") else "" for p in profiles]),
        ("Version", [
            p.get("eb_name", "").split("/", 1)[1].rsplit("-", 1)[0]
            if "/" in p.get("eb_name", "") and "-" in p.get("eb_name", "").split("/", 1)[1]
            else ""
            for p in profiles
        ]),
        ("Toolchain", [
            p.get("eb_name", "").split("/", 1)[1].rsplit("-", 1)[1]
            if "/" in p.get("eb_name", "") and "-" in p.get("eb_name", "").split("/", 1)[1]
            else ""
            for p in profiles
        ]),
        ("System", [p.get("system", "") for p in profiles]),
        ("NCPUs", [p.get("exp_config", {}).get("omp_num_threads", "") for p in profiles]),
        ("CB ID", [p.get("_cb_id", "") for p in profiles]),
        ("CPU", [p.get("cpu_model", "") for p in profiles]),
        ("N", [p.get("exp_config", {}).get("laab_n", "") for p in profiles]),
        ("REP", [p.get("exp_config", {}).get("laab_rep", "") for p in profiles]),
        ("Cutoff", [p.get("cutoff", "") for p in profiles]),
    ]

    common = []
    for label, values in candidates:
        if values and all(v == values[0] for v in values):
            common.append((label, values[0]))
    return common


def _all_same(values):
    return bool(values) and all(v == values[0] for v in values)


def _compare_sort_key(item):
    try:
        ncpus = int(item["selected"]["ncpus"])
    except (TypeError, ValueError, KeyError):
        ncpus = 10**9

    try:
        cb_id = int(item["selected"]["cb_id"])
    except (TypeError, ValueError, KeyError):
        cb_id = 10**9

    return (
        ncpus,
        item["selected"].get("library", ""),
        item["selected"].get("version", ""),
        item["selected"].get("toolchain", ""),
        cb_id,
    )


def _build_compare_groups(profiles):
    groups = []
    for title, experiments in COMPARE_GROUPS:
        rows = []
        for exp_key, row_label in experiments:
            values = []
            for profile in profiles:
                loss = profile["losses"][exp_key]["result"]
                result = profile["cutoff_results"][exp_key]["result"]
                values.append(
                    {
                        "loss": loss,
                        "result": result,
                        "symbol": _result_symbol(result),
                    }
                )
            rows.append({"label": row_label, "values": values})
        groups.append({"title": title, "rows": rows})
    return groups


def _build_sections(profile):
    sections = []

    for exp_name, exp_cfg in profile["config"].items():
        meta = EXPERIMENT_METADATA.get(exp_name, {})
        rows = []
        tests = exp_cfg.get("tests", {})
        times = profile["times"][exp_name]
        slow_down = profile["slow_down"][exp_name]
        losses = profile["losses"][exp_name]
        cutoff_results = profile["cutoff_results"][exp_name]

        for test_name, call in tests.items():
            rows.append(
                {
                    "label": meta.get("labels", {}).get(test_name, test_name),
                    "call": call,
                    "time": times["tests"].get(test_name, ""),
                    "slowdown": slow_down.get(test_name, ""),
                    "loss": losses.get(test_name, ""),
                    "result": cutoff_results.get(test_name, ""),
                }
            )

        rows.append(
            {
                "label": "<strong>Ref (-)</strong>",
                "call": meta.get("ref_negative_label", "reference (-)"),
                "time": times.get("ref_negative", ""),
                "slowdown": slow_down.get("ref_negative", ""),
                "loss": "",
                "result": "",
            }
        )
        rows.append(
            {
                "label": "<strong>Ref (+)</strong>",
                "call": exp_cfg.get("ref_positive", ""),
                "time": times.get("ref_positive", ""),
                "slowdown": "-",
                "loss": "",
                "result": "",
            }
        )

        sections.append(
            {
                "name": exp_name,
                "title": meta.get("title", exp_name),
                "expr_header": meta.get("expr_header", "Expr"),
                "operands": meta.get("operands", ""),
                "description": meta.get("description", ""),
                "result": _humanize_result(cutoff_results.get("result", "")),
                "rows": rows,
            }
        )

    return sections


@bp.route("/")
def index():
    profiles = PROFILES.get_profiles()
    return render_template(f"{name}/index.html", profiles=profiles)


@bp.route("/compare", methods=["POST"])
def compare():
    profile_paths = [p for p in request.form.get("profiles", "").split(",") if p]
    by_path = {row["profile_path"]: row for row in PROFILES.get_profiles()}
    pairs = []

    for profile_path in profile_paths:
        row = by_path.get(profile_path, {"profile_path": profile_path})
        profile = _load_profile(profile_path)
        profile["_cb_id"] = row.get("cb_id", _extract_cb_id(profile_path))
        pairs.append(
            {
                "profile": profile,
                "selected": {
                    "profile_path": profile_path,
                    "library": row.get("library", ""),
                    "version": row.get("version", ""),
                    "toolchain": row.get("toolchain", ""),
                    "system": row.get("system", ""),
                    "ncpus": row.get("omp_num_threads", ""),
                    "cb_id": row.get("cb_id", ""),
                },
            }
        )

    if not _all_same([pair["selected"]["ncpus"] for pair in pairs]):
        pairs.sort(key=_compare_sort_key)

    loaded_profiles = [pair["profile"] for pair in pairs]
    selected_profiles = [pair["selected"] for pair in pairs]

    show_ncpus_in_header = not _all_same([row["ncpus"] for row in selected_profiles])
    show_cb_id_in_header = not _all_same([row["cb_id"] for row in selected_profiles])
    show_system_in_header = not _all_same([row["system"] for row in selected_profiles])

    return render_template(
        f"{name}/compare.html",
        selected_profiles=selected_profiles,
        common_metadata=_build_common_metadata(loaded_profiles),
        compare_groups=_build_compare_groups(loaded_profiles),
        show_system_in_header=show_system_in_header,
        show_ncpus_in_header=show_ncpus_in_header,
        show_cb_id_in_header=show_cb_id_in_header,
        summary_rows=[
            {
                "label": "Mean loss",
                "values": [{"text": profile.get("mean_loss", "")} for profile in loaded_profiles],
            },
            {
                "label": "Score",
                "values": [
                    {"text": f"{profile.get('score', '')}/{profile.get('num_tests', '')}"}
                    for profile in loaded_profiles
                ],
            },
        ],
    )


@bp.route("/acknowledgements")
def acknowledgements():
    return render_template(f"{name}/acknowledgements.html")


@bp.route("/laab_report")
def laab_report():
    profile_path = request.args.get("profile_path")
    if not profile_path:
        return render_template(f"{name}/error.html", message="Profile path missing.", code=400), 400

    profile_file = PROFILES.profiles_root / profile_path
    if not profile_file.exists():
        return render_template(f"{name}/error.html", message="Profile not found.", code=404), 404

    try:
        with profile_file.open("rb") as handle:
            profile = pickle.load(handle)
    except Exception as exc:
        return render_template(
            f"{name}/error.html",
            message=f"Error loading profile: {exc}",
            code=500,
        ), 500

    return render_template(
        f"{name}/laab_report.html",
        profile=profile,
        profile_path=profile_path,
        cb_id=_extract_cb_id(profile_path),
        sections=_build_sections(profile),
    )
