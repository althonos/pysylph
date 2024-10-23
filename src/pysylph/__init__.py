from . import lib
from .lib import Sketcher, GenomeSketch, SequenceSketch, Database, DatabaseFile, AniResult, query, profile

__version__ = lib.__version__
__author__ = lib.__author__
__doc__ = lib.__doc__
# __build__ = lib.__build__
__all__ = [
    "Sketcher",
    "Database",
    "DatabaseFile",
    "GenomeSketch",
    "SequenceSketch",
    "AniResult",
    "query"
    "profile"
]
