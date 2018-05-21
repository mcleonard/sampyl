"""
sampyl.progressbar
~~~~~~~~~~~~~~~~~~~~

Progress bar for samplers.

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""

import sys

def update_progress(current, total, width=30, end=False):
    bar_width = width
    block = int(round(bar_width * current/total))
    text = "\rProgress: [{0}] {1} of {2} samples".\
             format("#"*block + "-"*(bar_width-block), current, total)
    if end:
        text = text +'\n'
    sys.stdout.write(text)
    sys.stdout.flush()
