"""Modulation formats for elastic optical networks.

Defines modulation formats with reach constraints and spectral efficiency,
used for Agent-R's modulation selection and FS demand calculation.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ModulationFormat:
    """A modulation format in EON."""
    name: str              # e.g., "BPSK", "QPSK", "8QAM", "16QAM"
    reach_km: float        # Maximum transmission distance (km)
    spectral_efficiency: float  # bits/s/Hz
    # Common values:
    #   BPSK:  reach=4000km, SE=1.0
    #   QPSK:  reach=2000km, SE=2.0
    #   8QAM:  reach=1000km, SE=3.0
    #   16QAM: reach=500km,  SE=4.0


# Default modulation table (can be overridden via config)
DEFAULT_MODULATIONS: List[ModulationFormat] = [
    ModulationFormat("BPSK",  4000.0, 1.0),
    ModulationFormat("QPSK",  2000.0, 2.0),
    ModulationFormat("8QAM",  1000.0, 3.0),
    ModulationFormat("16QAM",  500.0, 4.0),
]

# Extended modulation table with finer SE gradations and PM-BPSK for ultra-long reach.
# SE increases as reach decreases, maintaining inverse relationship.
EXTENDED_MODULATIONS: List[ModulationFormat] = [
    ModulationFormat("PM-BPSK", 8000.0, 1.0),
    ModulationFormat("BPSK",    4000.0, 1.0),
    ModulationFormat("QPSK",    2000.0, 2.0),
    ModulationFormat("8QAM",    1000.0, 3.0),
    ModulationFormat("16QAM",    500.0, 4.0),
    ModulationFormat("32QAM",    250.0, 5.0),
    ModulationFormat("64QAM",    125.0, 6.0),
]


def get_feasible_modulations(path_length_km: float,
                              mod_table: Optional[List[ModulationFormat]] = None
                              ) -> List[ModulationFormat]:
    """Return modulation formats whose reach >= path_length_km.

    Args:
        path_length_km: Physical length of the path (km).
        mod_table: Custom modulation table. Uses DEFAULT_MODULATIONS if None.

    Returns:
        List of feasible ModulationFormat, sorted by spectral_efficiency desc.
    """
    table = mod_table if mod_table is not None else DEFAULT_MODULATIONS
    feasible = [m for m in table if m.reach_km >= path_length_km]
    feasible.sort(key=lambda m: m.spectral_efficiency, reverse=True)
    return feasible


def get_modulation_by_name(name: str,
                           mod_table: Optional[List[ModulationFormat]] = None
                           ) -> Optional[ModulationFormat]:
    """Look up a modulation format by name."""
    table = mod_table if mod_table is not None else DEFAULT_MODULATIONS
    for m in table:
        if m.name == name:
            return m
    return None


class ModulationRegistry:
    """Registry for modulation formats, supports custom configs."""

    def __init__(self, mod_table: Optional[List[ModulationFormat]] = None):
        self.mod_table = list(mod_table) if mod_table is not None else list(DEFAULT_MODULATIONS)
        self._name_to_idx = {m.name: i for i, m in enumerate(self.mod_table)}

    @classmethod
    def from_profile(cls, profile: str = "default") -> "ModulationRegistry":
        """Create a registry from a named profile.

        Args:
            profile: "default" (4 formats) or "extended" (7 formats with PM-BPSK).

        Returns:
            ModulationRegistry with the requested modulation table.
        """
        profiles = {
            "default": DEFAULT_MODULATIONS,
            "extended": EXTENDED_MODULATIONS,
        }
        mod_table = profiles.get(profile, DEFAULT_MODULATIONS)
        return cls(mod_table=mod_table)

    @property
    def num_formats(self) -> int:
        return len(self.mod_table)

    @property
    def names(self) -> List[str]:
        return [m.name for m in self.mod_table]

    def __getitem__(self, idx: int) -> ModulationFormat:
        return self.mod_table[idx]

    def by_name(self, name: str) -> Optional[ModulationFormat]:
        return get_modulation_by_name(name, self.mod_table)

    def feasible_for_path(self, path_length_km: float) -> List[ModulationFormat]:
        return get_feasible_modulations(path_length_km, self.mod_table)

    def index_of(self, name: str) -> int:
        return self._name_to_idx[name]
