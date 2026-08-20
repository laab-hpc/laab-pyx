import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ProfilesIndex:
    def __init__(self, profiles_root):
        self.profiles_root = Path(profiles_root).expanduser()
        self.profiles_file = None
        self.profiles = []
        self.version = None

        self._load_profiles()

    def _load_profiles(self):
        self.profiles_file = next(self.profiles_root.glob("profiles.*.json"), None)

        if self.profiles_file is None:
            logger.error("Could not find profiles.*.json file in %s", self.profiles_root)
            raise FileNotFoundError(
                f"Could not find profiles.*.json file in {self.profiles_root}"
            )

        self.version = self.profiles_file.name.split(".")[1]
        with self.profiles_file.open("r", encoding="utf-8") as handle:
            self.profiles = json.load(handle)
        logger.info("[IO] Loaded %d profiles from %s", len(self.profiles), self.profiles_file)

    def get_profiles(self):
        if self.profiles_file.exists():
            return self.profiles

        self._load_profiles()
        return self.profiles


PROFILES = ProfilesIndex(os.environ["PYX_PROFILES_ROOT"])
