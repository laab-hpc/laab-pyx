import argparse
from datetime import datetime
import json
import pickle
from pathlib import Path


def _split_framework(eb_name: str) -> tuple[str, str, str]:
    if "/" not in eb_name:
        return eb_name, "", ""

    library, version_toolchain = eb_name.split("/", 1)
    if "-" not in version_toolchain:
        return library, version_toolchain, ""

    version, toolchain = version_toolchain.split("-", 1)
    return library, version, toolchain


def _extract_cb_id(profile_path: Path) -> str:
    stem_parts = profile_path.stem.split("_")
    if len(stem_parts) < 3:
        return ""
    return stem_parts[-1]


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
        "name": exp_config.get("name", ""),
        "laab_n": exp_config.get("laab_n", ""),
        "laab_rep": exp_config.get("laab_rep", ""),
        "omp_num_threads": exp_config.get("omp_num_threads", ""),
        "datetime": exp_config.get("datetime", ""),
        "cb_id": _extract_cb_id(profile_path),
    }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LAAB-PyX profile metadata")
    parser.add_argument("profiles_root", help="Directory under which profiles/profile_*.pkl files are searched")
    args = parser.parse_args()

    profiles_root = Path(args.profiles_root).expanduser().resolve()
    if not profiles_root.exists():
        raise FileNotFoundError(f"Profiles root {profiles_root} not found")

    profile_paths = sorted(
        p.resolve() for p in profiles_root.rglob("profiles/profile_*.pkl") if p.is_file()
    )

    profiles = [_load_profile_metadata(path, profiles_root) for path in profile_paths]
    profiles.sort(
        key=lambda item: (
            item["name"],
            item["system"],
            item["library"],
            item["version"],
            item["toolchain"],
            str(item["omp_num_threads"]),
            str(item["cb_id"]),
        )
    )

    output_dir = profiles_root
    old_files = list(output_dir.glob("profiles.*.json"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"profiles.{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, indent=4, default=str)

    for old_file in old_files:
        old_file.unlink()

    print(f"Wrote {len(profiles)} profiles to {output_path}")


if __name__ == "__main__":
    main()
