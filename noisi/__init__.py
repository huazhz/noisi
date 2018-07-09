import sys
import mpi4py
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
print('='*80)
print("NOISI toolkit")
print("Python version: "+sys.version)
print("mpi4py version: "+mpi4py.__version__)
print(_ROOT)
print('='*80)
from .my_classes.wavefield import WaveField
from .my_classes.noisesource import NoiseSource
from .my_classes.basisfunction import BasisFunction

