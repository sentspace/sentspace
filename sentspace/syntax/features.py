import pandas as pd
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


class Feature:
    def __init__(self):
        self.tree = None
        self.dlt = None
        self.left_corner = None


class Tree:
    """Description of a Tree object"""

    def __init__(self, data=None):
        self.raw = data

    def __repr__(self):
        return repr(self.raw)


class DLT:
    """Description of a DLT object"""

    def __init__(self, data=None):
        self.pandas_dataframe = pd.read_csv(StringIO(data), sep=' ')
        self.raw = data

    def __repr__(self):
        return repr(self.pandas_dataframe)


class LeftCorner:
    """Description of a LeftCorner object"""

    def __init__(self, data=None):
        self.pandas_dataframe = pd.read_csv(StringIO(data), sep=' ')
        self.raw = data

    def __repr__(self):
        return repr(self.pandas_dataframe)