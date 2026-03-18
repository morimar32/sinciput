"""
Domain hierarchy loader.

Loads the 3-tier domain taxonomy (archipelago > island > domain) from domains.json
and provides a flat indexed list for use across all pipeline stages.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


DOMAINS_FILE = Path(__file__).parent.parent / "domains.json"


@dataclass(frozen=True)
class Domain:
    """A single domain with its position in the hierarchy."""
    index: int
    id: str                # snake_case, used for filenames and data IDs
    name: str              # human-readable, used in LLM prompts
    island: str            # parent grouping (e.g. "Computer Science")
    archipelago: str       # top-level grouping (e.g. "Applied Science")


def _to_snake_case(name: str) -> str:
    """Convert a human-readable domain name to a snake_case ID.

    Examples:
        "Cloud Computing" -> "cloud_computing"
        "3D Printing" -> "3d_printing"
        "IoT & Embedded Systems" -> "iot_and_embedded_systems"
        "Anatomy - Cardiovascular Anatomy" -> "anatomy_cardiovascular_anatomy"
        "Botany - Ferns & Bryophytes" -> "botany_ferns_and_bryophytes"
    """
    s = name.lower()
    s = s.replace(" - ", "_")
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def load_domains(path: Path = DOMAINS_FILE) -> list[Domain]:
    """Load domains from the JSON hierarchy file.

    Returns a flat list of Domain objects with stable integer indices.
    The ordering is deterministic: archipelagos, islands, and domains
    are iterated in the order they appear in the JSON file.
    """
    with open(path) as f:
        data = json.load(f)

    domains = []
    index = 0
    for arch in data["archipelagos"]:
        for island in arch["islands"]:
            for name in island["domains"]:
                domains.append(Domain(
                    index=index,
                    id=_to_snake_case(name),
                    name=name,
                    island=island["name"],
                    archipelago=arch["name"],
                ))
                index += 1

    # Sanity check for duplicate IDs
    ids = [d.id for d in domains]
    dupes = [id for id in ids if ids.count(id) > 1]
    if dupes:
        raise ValueError(f"Duplicate domain IDs: {set(dupes)}")

    return domains


# Module-level singleton for convenience
DOMAINS = load_domains()
DOMAIN_IDS = [d.id for d in DOMAINS]
DOMAIN_BY_ID = {d.id: d for d in DOMAINS}
