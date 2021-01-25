"""TrafficJamPrediction module.
Exposes:
* :py:mod:`~trafficjamprediction.__version__` number

"""

# TODO: Write a nicer summary.

import sys
sys.path.append(".")

from ._version import __version__

from .trafficJamPrediction import *

from . import preprocessing
from . import ai
from . import plotting
from . import utils
from . import io
