from . import lib
from .lib import Sketcher, GenomeSketch, SequenceSketch, Database, AniResult, query

__version__ = lib.__version__
__author__ = lib.__author__
__doc__ = lib.__doc__
# __build__ = lib.__build__
__all__ = [
    "Sketcher",
    "Database",
    "GenomeSketch",
    "SequenceSketch",
    "AniResult",
    "query"
]
