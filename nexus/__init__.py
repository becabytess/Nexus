"""Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine.

A graph-native knowledge architecture where concepts are standalone Markdown documents,
connections point UPWARD into higher-level applications, memory decays continuously
with structural anchoring, and mastery propagates asymmetrically downward.
"""

from nexus.engine import (
    NexusEngine,
    breadth_multiplier,
    decay_factor,
    slugify,
    update_half_life,
)
from nexus.models import (
    ConceptBlueprint,
    Connection,
    ImplicitReinforcement,
    ReviewResult,
    StudyCandidate,
)
from nexus.storage import DEFAULT_GRAPH_FILE, export_markdown, load_graph, save_graph
from nexus.tutor import SocraticTutor

__all__ = [
    "NexusEngine",
    "ConceptBlueprint",
    "Connection",
    "ImplicitReinforcement",
    "ReviewResult",
    "StudyCandidate",
    "SocraticTutor",
    "slugify",
    "breadth_multiplier",
    "decay_factor",
    "update_half_life",
    "save_graph",
    "load_graph",
    "export_markdown",
    "DEFAULT_GRAPH_FILE",
]

__version__ = "0.3.0"