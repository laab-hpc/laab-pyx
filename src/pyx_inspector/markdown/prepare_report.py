import argparse
import os
import pickle
from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader, StrictUndefined


TEST_TEMPLATES = {
    "Test-1-Matmul": "test_1_matmul.md",
    "Test-2-CSE": "test_2_cse.md",
    "Test-3-Matrix-Chains": "test_3_matrix_chains.md",
    "Test-4-Expression-Rewrites": "test_4_expression_rewrites.md",
    "Test-5-Code-Motion": "test_5_code_motion.md",
}


def _safe_component(value: object, field: str) -> str:
    component = str(value).strip()
    separators = {os.sep}
    if os.altsep:
        separators.add(os.altsep)

    if (
        not component
        or component in {".", ".."}
        or any(separator in component for separator in separators)
    ):
        raise ValueError(f"Invalid {field} path component: {component!r}")
    return component


def _split_framework(eb_name: object) -> tuple[str, str]:
    value = str(eb_name).strip()
    if value.count("/") != 1:
        raise ValueError(
            f"Expected eb_name in library/version-toolchain form, got {value!r}"
        )
    library, version_toolchain = value.split("/", 1)
    return (
        _safe_component(library, "library"),
        _safe_component(version_toolchain, "version-toolchain"),
    )


def discover_profiles(profiles_root: Path) -> list[Path]:
    profiles_root = profiles_root.expanduser().resolve()
    if not profiles_root.is_dir():
        raise NotADirectoryError(f"Profiles root {profiles_root} not found")

    profile_paths = set(profiles_root.rglob("profiles/*.pkl"))
    if profiles_root.name == "profiles":
        profile_paths.update(profiles_root.glob("*.pkl"))

    profiles = sorted(path.resolve() for path in profile_paths if path.is_file())
    if not profiles:
        raise FileNotFoundError(
            f"No profiles/*.pkl files found under {profiles_root}"
        )
    return profiles


def _load_profile(profile_path: Path) -> dict:
    with profile_path.open("rb") as handle:
        profile = pickle.load(handle)
    if not isinstance(profile, dict):
        raise TypeError(f"Expected a pickled dict in {profile_path}")
    return profile


def _report_name(profile: dict, profile_path: Path) -> str:
    cb_match = re.search(r"[-_](\d+)$", profile_path.stem)
    if cb_match is None:
        raise ValueError(f"Cannot determine CB ID from {profile_path.name!r}")

    try:
        cb_id = int(cb_match.group(1))
        ncpus = int(profile.get("exp_config", {})["omp_num_threads"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot determine CPU count from {profile_path}") from exc

    if cb_id < 0 or ncpus < 1:
        raise ValueError(
            f"Invalid CB ID or CPU count in {profile_path}: {cb_id}, {ncpus}"
        )
    return f"{cb_id}--{ncpus}"


def _output_path(
    profile: dict, profile_path: Path, out_dir: Path, report_name: str
) -> Path:
    library, version_toolchain = _split_framework(profile.get("eb_name", ""))
    system = _safe_component(profile.get("system", ""), "system")
    test_id = _safe_component(profile.get("test_id", ""), "test ID")
    return (
        out_dir
        / library
        / version_toolchain
        / system
        / test_id
        / f"{_safe_component(report_name, 'report name')}.md"
    )


def _legacy_output_path(
    profile: dict, profile_path: Path, out_dir: Path
) -> Path:
    library, version_toolchain = _split_framework(profile.get("eb_name", ""))
    system = _safe_component(profile.get("system", ""), "system")
    legacy_name = _safe_component(profile_path.stem, "profile filename") + ".md"
    return out_dir / library / version_toolchain / system / legacy_name


def render_profiles(profiles_root: Path, out_dir: Path) -> list[Path]:
    templates_dir = Path(__file__).resolve().parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(templates_dir),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )

    out_dir = out_dir.expanduser().resolve()
    rendered_paths = []

    for profile_path in discover_profiles(profiles_root):
        profile = _load_profile(profile_path)
        test_id = str(profile.get("test_id", ""))
        template_name = TEST_TEMPLATES.get(test_id)
        if template_name is None:
            raise ValueError(
                f"Unknown test_id {test_id!r} in {profile_path}; expected one of "
                f"{sorted(TEST_TEMPLATES)}"
            )

        report_name = _report_name(profile, profile_path)
        report_path = _output_path(profile, profile_path, out_dir, report_name)
        legacy_path = _legacy_output_path(profile, profile_path, out_dir)
        if legacy_path.is_file():
            legacy_path.unlink()
            print(f"Removed obsolete {legacy_path}")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        existed = report_path.exists()
        context = dict(profile)
        context.update(
            {
                "report_name": report_name,
                "cb_id": report_name.split("--", 1)[0],
            }
        )
        report = environment.get_template(template_name).render(**context)
        report_path.write_text(report, encoding="utf-8")
        rendered_paths.append(report_path)
        action = "Replaced" if existed else "Wrote"
        print(f"{action} {report_path}")

    return rendered_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively render one test-specific Markdown report for every "
            "profiles/*.pkl file"
        )
    )
    parser.add_argument(
        "profiles_root",
        type=Path,
        help="Directory recursively containing profiles/*.pkl files",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        "--out_dir",
        type=Path,
        required=True,
        help="Root directory for framework/version-toolchain/system reports",
    )
    args = parser.parse_args()

    reports = render_profiles(args.profiles_root, args.out_dir)
    print(f"Rendered {len(reports)} Markdown reports")


if __name__ == "__main__":
    main()

