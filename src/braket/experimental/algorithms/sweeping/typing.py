from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

UNITARY_BLOCK: TypeAlias = tuple[list[int], NDArray[np.complex128]]
UNITARY_LAYER: TypeAlias = list[UNITARY_BLOCK]
