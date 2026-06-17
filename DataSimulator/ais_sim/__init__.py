"""Synthetic maritime AIS data simulator.

Generates plausible "normal" vessel traffic for a configurable geographic
region (default: the wider Caribbean) and injects a tunable, low rate of
labeled anomalous behavior for anomaly-detection R&D.

This is a *data simulator*, not a maritime physics engine. The goal is to
produce data with realistic, *detectable* anomaly signatures plus quasi
ground-truth labels so a data science team can start building and evaluating
detectors before real commercial AIS data is procured.

Anomaly types modeled (grounded in published MDA / IUU-fishing literature):
    - illegal_fishing          slow loitering inside a restricted zone
    - transshipment            two vessels rendezvous at sea (<500m, <2kn, far from port)
    - ais_spoofing             false position / teleportation (impossible speed jump)
    - dark_activity            vessel goes dark (AIS gap), reappears displaced
    - aggressive_maneuvering   one vessel shadows/harasses another at close range
"""

from .config import Config, ANOMALY_TYPES
from .simulator import simulate

__all__ = ["Config", "ANOMALY_TYPES", "simulate"]
__version__ = "0.1.0"
