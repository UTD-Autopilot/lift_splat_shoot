# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import models.pidnet as pidnet
except ModuleNotFoundError as e:
    import pidnet.models.pidnet as pidnet