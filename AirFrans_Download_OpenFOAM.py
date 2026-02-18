import airfrans as af
import numpy as np
from pathlib import Path



af.download(root=Path("./afOpenFOAM"), filename="afOpenFOAM", unzip=True, OpenFOAM=True)