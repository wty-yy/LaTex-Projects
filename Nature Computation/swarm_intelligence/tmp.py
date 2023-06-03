import numpy as np
from pathlib import Path
PATH_DATA = Path(__file__).parent
tmp = np.loadtxt(PATH_DATA.joinpath("tmp.txt")).T
np.savetxt(PATH_DATA.joinpath("tmpT.txt"), tmp, fmt="%.2f")