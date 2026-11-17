"""Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine.

A graph-native knowledge architecture where concepts are rich personal mental models,
connections point UPWARD into higher-level applications, and selection scans upward
to identify frayed connections holding up complex cognitive structures.
"""

from nexus.engine import NexusEngine, slugify
from nexus.models import (
    CheckpointSpec,
    ConceptBlueprint,
    Connection,
    StudyCandidate,
)
from nexus.storage import DEFAULT_GRAPH_FILE, load_graph, save_graph
from nexus.tutor import SocraticTutor

__all__ = [
    "NexusEngine",
    "ConceptBlueprint",
    "Connection",
    "StudyCandidate",
    "CheckpointSpec",
    "SocraticTutor",
    "slugify",
    "save_graph",
    "load_graph",
    "DEFAULT_GRAPH_FILE",
]

__version__ = "0.1.0"
