import abc
import typing
from os import PathLike
from typing import List, Sequence, Iterable, Iterator, BinaryIO, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

K = Literal[21, 31]
SEQUENCE = Union[str, bytes, bytearray, memoryview]

class Sketch(abc.ABC):
    @property
    @abc.abstractmethod
    def c(self) -> int: ...
    @property
    @abc.abstractmethod
    def k(self) -> K: ...

class GenomeSketch(Sketch):
    @property
    def k(self) -> K: ...
    @property
    def c(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def genome_size(self) -> int: ...
    @property
    def min_spacing(self) -> int: ...
    @property
    def kmers(self) -> List[int]: ...

class Database(Sequence[GenomeSketch]):
    def __init__(self, items: Iterable[GenomeSketch] = ()) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> GenomeSketch: ...
    @classmethod
    def load(cls, file: Union[str, PathLike, BinaryIO]) -> Database: ...
    def dump(self, path: Union[str, PathLike]) -> None: ...

class DatabaseReader(Iterator[GenomeSketch]):
    def __init__(self, path: Union[str, PathLike]) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DatabaseReader: ...
    def __next__(self) -> GenomeSketch: ...

class SampleSketch(Sketch):
    @property
    def k(self) -> K: ...
    @property
    def c(self) -> int: ...
    @classmethod
    def load(cls, file: Union[str, PathLike, BinaryIO]) -> SampleSketch: ...
    def dump(self, path: Union[str, PathLike]) -> None: ...

class AniResult:
    @property
    def genome_sketch(self) -> GenomeSketch: ...
    @property
    def ani(self) -> float: ...
    @property
    def ani_naive(self) -> float: ...
    @property
    def coverage(self) -> float: ...

class ProfileResult(AniResult):
    @property
    def sequence_abundance(self) -> float: ...
    @property
    def taxonomic_abundance(self) -> float: ...
    @property
    def kmers_reassigned(self) -> int: ...

class Sketcher:
    def __init__(self, c: int = 200, k: K = 31) -> None: ...
    def sketch_genome(
        self, name: str, contigs: Iterable[SEQUENCE], profiling: bool = True
    ) -> GenomeSketch: ...
    def sketch_single(self, name: str, reads: Iterable[SEQUENCE]) -> SampleSketch: ...
    def sketch_paired(self, name: str, r1: Iterable[SEQUENCE], r2: Iterable[SEQUENCE]) -> SampleSketch: ...

class Profiler:
    def __init__(
        self,
        database: Database,
        *,
        minimum_ani: Optional[float],
        seq_id: Optional[float],
        estimate_unknown: bool = False,
        min_number_kmers: int = 50,
    ) -> None: ...
    def query(self, sample: SampleSketch) -> List[AniResult]: ...
    def profile(self, sample: SampleSketch) -> List[ProfileResult]: ...
