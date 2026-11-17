import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import tempfile

from ._laab_results import LAABResults


_METADATA_FIELDS = (
    "test_id",
    "name",
    "system",
    "library",
    "version",
    "toolchain",
)


def format_floats_recursive(data: dict, precision: int = 2) -> dict:
    """Recursively format float values to the requested precision."""
    for key, value in data.items():
        if isinstance(value, float):
            data[key] = float(f"{value:.{precision}f}")
        elif isinstance(value, dict):
            format_floats_recursive(value, precision)
    return data


def format_cutoff_results_md(cutoff_results: dict) -> dict:
    for experiment, tests in cutoff_results.items():
        for test, result in tests.items():
            cutoff_results[experiment][test] = (
                ":white_check_mark:" if result else ":x:"
            )
    return cutoff_results


def _require_logs_dir(logs_dir: Path) -> None:
    if not logs_dir.is_dir():
        raise NotADirectoryError(f"Logs directory {logs_dir} not found")

    required_files = ("data.txt", "cpu_info.txt", "metadata.txt")
    missing = [name for name in required_files if not (logs_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Logs directory {logs_dir} is missing: {', '.join(missing)}"
        )


def _read_metadata(metadata_file: Path) -> dict:
    lines = [
        line.strip()
        for line in metadata_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < len(_METADATA_FIELDS):
        raise ValueError(
            f"Metadata file {metadata_file} must contain at least "
            f"{len(_METADATA_FIELDS)} non-empty lines"
        )
    return dict(zip(_METADATA_FIELDS, lines))


def _read_run_configuration(data_file: Path) -> tuple[str, int]:
    repetitions = set()
    problem_sizes = set()

    for line in data_file.read_text(encoding="utf-8").splitlines():
        rep_match = re.search(r"\bREP=(\d+)\b", line)
        if rep_match:
            repetitions.add(int(rep_match.group(1)))

        size_match = re.search(r"\.exe\s+(\d+)(?:\s|$)", line)
        if size_match:
            problem_sizes.add(size_match.group(1))

    if len(repetitions) != 1:
        raise ValueError(
            f"Expected one REP value in {data_file}, found {sorted(repetitions)}"
        )
    if len(problem_sizes) != 1:
        raise ValueError(
            f"Expected one problem size in {data_file}, found "
            f"{sorted(problem_sizes)}"
        )

    return problem_sizes.pop(), repetitions.pop()


def _read_num_cpus(name: str) -> int:
    match = re.fullmatch(r"CPU-(\d+)x", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine CPU count from experiment name {name!r}")
    return int(match.group(1))


def _read_cb_id(logs_dir: Path) -> str:
    for ancestor in logs_dir.parents:
        if re.fullmatch(r"\d+_[^/]+", ancestor.name):
            benchmark_dir = ancestor.parent
            if re.fullmatch(r"\d+", benchmark_dir.name):
                return str(int(benchmark_dir.name))
    raise ValueError(f"Cannot determine JUBE benchmark ID from {logs_dir}")


def _read_config(logs_dir: Path) -> dict:
    config_files = sorted(logs_dir.glob("*.json"))
    if not config_files:
        raise FileNotFoundError(f"No JSON configuration files found in {logs_dir}")

    config = {}
    for config_file in config_files:
        with config_file.open("r", encoding="utf-8") as handle:
            fragment = json.load(handle)
        if not isinstance(fragment, dict):
            raise TypeError(f"Expected a JSON object in {config_file}")
        duplicate = set(config) & set(fragment)
        if duplicate:
            raise ValueError(
                f"Duplicate experiments in {config_file}: {sorted(duplicate)}"
            )
        config.update(fragment)
    return config


def _profile_filename(test_id: str, num_cpus: int, cb_id: str) -> str:
    test_slug = re.sub(r"[^a-z0-9]+", "-", test_id.lower()).strip("-")
    if not test_slug:
        raise ValueError(f"Cannot create a filename from test ID {test_id!r}")
    return f"{test_slug}-{num_cpus}-{cb_id}.pkl"


def dump_results_pickle(
    laab_results,
    config: dict,
    exp_config: dict,
    test_id: str,
    pickle_path: Path,
    cutoff: float = 0.05,
) -> None:
    if set(laab_results.data) != set(config):
        missing_in_data = set(config) - set(laab_results.data)
        missing_in_config = set(laab_results.data) - set(config)
        raise ValueError(
            "Experiments in data and config do not match; "
            f"missing in data={sorted(missing_in_data)}, "
            f"missing in config={sorted(missing_in_config)}"
        )

    min_exec_times = laab_results.get_min_test_times()
    laab_results.compute_loss()
    losses = laab_results.loss
    mean_loss = laab_results.mean_loss
    cutoff_data = laab_results.apply_cutoff(cutoff=cutoff)

    precision = 3
    data = {
        "test_id": test_id,
        "eb_name": laab_results.eb_version,
        "system": laab_results.system,
        "cpu_model": laab_results.cpu_model,
        "losses": format_floats_recursive(losses, precision),
        "slow_down": format_floats_recursive(
            laab_results.slow_down, precision
        ),
        "mean_loss": f"{mean_loss:.{precision}f}",
        "cutoff_results": format_cutoff_results_md(cutoff_data.results),
        "score": cutoff_data.score,
        "num_tests": len(losses),
        "times": format_floats_recursive(min_exec_times, precision),
        "cutoff": f"{cutoff:.2f}",
        "config": config,
        "exp_config": exp_config,
    }

    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as handle:
        pickle.dump(data, handle)
    print(f"Results dumped to {pickle_path}")


def process_logs(logs_dir: Path, out_dir: Path) -> Path:
    logs_dir = logs_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    _require_logs_dir(logs_dir)

    metadata = _read_metadata(logs_dir / "metadata.txt")
    problem_size, repetitions = _read_run_configuration(logs_dir / "data.txt")
    num_cpus = _read_num_cpus(metadata["name"])
    cb_id = _read_cb_id(logs_dir)
    config = _read_config(logs_dir)

    combined_log = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as handle:
            combined_log = Path(handle.name)
            handle.write((logs_dir / "cpu_info.txt").read_text(encoding="utf-8"))
            handle.write("\n")
            handle.write((logs_dir / "data.txt").read_text(encoding="utf-8"))

        eb_name = (
            f"{metadata['library']}/{metadata['version']}-"
            f"{metadata['toolchain']}"
        )
        laab_results = LAABResults(combined_log, eb_name, metadata["system"])
    finally:
        if combined_log is not None:
            combined_log.unlink(missing_ok=True)

    exp_config = {
        "laab_n": problem_size,
        "laab_rep": repetitions,
        "name": metadata["name"],
        "omp_num_threads": num_cpus,
        "datetime": datetime.now().isoformat(timespec="seconds"),
    }
    output_path = out_dir / _profile_filename(
        metadata["test_id"], num_cpus, cb_id
    )
    dump_results_pickle(
        laab_results,
        config,
        exp_config,
        metadata["test_id"],
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a LAAB-PyX profile from a self-contained logs directory"
    )
    parser.add_argument("logs_dir", type=Path, help="Path to the logs directory")
    parser.add_argument(
        "-o",
        "--out-dir",
        "--out_dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for the automatically named profile (default: cwd)",
    )
    args = parser.parse_args()

    process_logs(args.logs_dir, args.out_dir)


if __name__ == "__main__":
    main()
