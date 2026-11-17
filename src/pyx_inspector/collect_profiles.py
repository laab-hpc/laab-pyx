import argparse
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
import re


def _split_framework(eb_name: str) -> tuple[str, str, str]:
    if "/" not in eb_name:
        return eb_name, "", ""

    library, version_toolchain = eb_name.split("/", 1)
    if "-" not in version_toolchain:
        return library, version_toolchain, ""

    version, toolchain = version_toolchain.split("-", 1)
    return library, version, toolchain


def _extract_cb_id(profile_path: Path) -> str:
    match = re.search(r"[-_](\d+)$", profile_path.stem)
    return match.group(1) if match else ""


def _load_profile_metadata(profile_path: Path, profiles_root: Path) -> dict:
    with profile_path.open("rb") as handle:
        profile = pickle.load(handle)

    if not isinstance(profile, dict):
        raise TypeError(f"Expected pickled dict in {profile_path}")

    eb_name = str(profile.get("eb_name", ""))
    library, version, toolchain = _split_framework(eb_name)
    exp_config = profile.get("exp_config", {})
    artifact_dir = profile_path.parent.parent

    metadata = {
        "profile_path": str(profile_path.relative_to(profiles_root)),
        "artifact_path": str(artifact_dir.relative_to(profiles_root)),
        "eb_name": eb_name,
        "library": library,
        "version": version,
        "toolchain": toolchain,
        "system": profile.get("system", ""),
        "cpu_model": profile.get("cpu_model", ""),
        "mean_loss": profile.get("mean_loss", ""),
        "score": profile.get("score", ""),
        "num_tests": profile.get("num_tests", ""),
        "cutoff": profile.get("cutoff", ""),
        "test_id": profile.get("test_id", ""),
        "name": exp_config.get("name", ""),
        "laab_n": exp_config.get("laab_n", ""),
        "laab_rep": exp_config.get("laab_rep", ""),
        "omp_num_threads": exp_config.get("omp_num_threads", ""),
        "datetime": exp_config.get("datetime", ""),
        "cb_id": _extract_cb_id(profile_path),
    }
    return metadata


def _group_key(metadata: dict) -> tuple:
    return (
        metadata["eb_name"],
        metadata["system"],
        metadata["cpu_model"],
        metadata["name"],
        str(metadata["laab_n"]),
        str(metadata["laab_rep"]),
        str(metadata["omp_num_threads"]),
        str(metadata["cb_id"]),
        str(metadata["cutoff"]),
    )


def _report_id(group_key: tuple) -> str:
    serialized = json.dumps(group_key, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _group_profiles(profiles: list[dict]) -> list[dict]:
    grouped = {}

    for profile in profiles:
        key = _group_key(profile)
        test_id = profile["test_id"] or Path(profile["profile_path"]).stem
        if key not in grouped:
            grouped[key] = {
                field: profile[field]
                for field in (
                    "eb_name",
                    "library",
                    "version",
                    "toolchain",
                    "system",
                    "cpu_model",
                    "cutoff",
                    "name",
                    "laab_n",
                    "laab_rep",
                    "omp_num_threads",
                    "cb_id",
                )
            }
            grouped[key].update(
                {
                    "report_id": _report_id(key),
                    "profile_paths": {},
                    "artifact_paths": {},
                    "score": 0,
                    "num_tests": 0,
                    "datetime": profile["datetime"],
                    "_weighted_loss": 0.0,
                }
            )

        group = grouped[key]
        if test_id in group["profile_paths"]:
            raise ValueError(
                f"Duplicate {test_id!r} profiles for report {group['report_id']}: "
                f"{group['profile_paths'][test_id]} and {profile['profile_path']}"
            )

        group["profile_paths"][test_id] = profile["profile_path"]
        group["artifact_paths"][test_id] = profile["artifact_path"]
        group["score"] += int(profile["score"])
        num_tests = int(profile["num_tests"])
        group["num_tests"] += num_tests
        group["_weighted_loss"] += float(profile["mean_loss"]) * num_tests
        group["datetime"] = max(group["datetime"], profile["datetime"])

    result = []
    for group in grouped.values():
        group["test_ids"] = sorted(group["profile_paths"])
        group["profile_paths"] = {
            test_id: group["profile_paths"][test_id]
            for test_id in group["test_ids"]
        }
        group["artifact_paths"] = {
            test_id: group["artifact_paths"][test_id]
            for test_id in group["test_ids"]
        }
        if group["num_tests"]:
            group["mean_loss"] = (
                f"{group.pop('_weighted_loss') / group['num_tests']:.3f}"
            )
        else:
            group.pop("_weighted_loss")
            group["mean_loss"] = "0.000"
        result.append(group)

    result.sort(
        key=lambda item: (
            item["name"],
            item["system"],
            item["library"],
            item["version"],
            item["toolchain"],
            int(item["omp_num_threads"]),
            int(item["cb_id"]),
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LAAB-PyX profile metadata")
    parser.add_argument(
        "profiles_root",
        help="Directory under which profiles/*.pkl files are searched",
    )
    args = parser.parse_args()

    profiles_root = Path(args.profiles_root).expanduser().resolve()
    if not profiles_root.exists():
        raise FileNotFoundError(f"Profiles root {profiles_root} not found")

    profile_paths = sorted(
        p.resolve() for p in profiles_root.rglob("profiles/*.pkl") if p.is_file()
    )

    profiles = [_load_profile_metadata(path, profiles_root) for path in profile_paths]
    profile_groups = _group_profiles(profiles)

    output_dir = profiles_root
    old_files = list(output_dir.glob("profiles.*.json"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"profiles.{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(profile_groups, handle, indent=4, default=str)

    for old_file in old_files:
        old_file.unlink()

    print(
        f"Wrote {len(profile_groups)} report groups from "
        f"{len(profiles)} profiles to {output_path}"
    )


if __name__ == "__main__":
    main()
