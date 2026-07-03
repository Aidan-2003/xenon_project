import sys

plugins_path = "/mnt/c/Users/Aidan/DataspellProjects/xenon_project/conor_plugins"
if plugins_path not in sys.path:
    sys.path.insert(0, plugins_path)

from count_ne import PiecewiseInfo, CountNElectron
from SubTyping_Class import PeaksSubtypes
from Wrong_pS2_relabel import PS2_relabel

def register_conor_plugins(straxen_context):
    for plugin in [PiecewiseInfo, CountNElectron, PeaksSubtypes, PS2_relabel]:
        straxen_context.register(plugin)
    return straxen_context
