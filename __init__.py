import os
import shutil

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

_temp_dir = os.path.join(os.path.dirname(__file__), "temp_convos")
try:
    if os.path.isdir(_temp_dir):
        shutil.rmtree(_temp_dir, ignore_errors=True)
    os.makedirs(_temp_dir, exist_ok=True)
except Exception:
    pass

_temp_images = os.path.join(os.path.dirname(__file__), "temp_images")
try:
    if os.path.isdir(_temp_images):
        shutil.rmtree(_temp_images, ignore_errors=True)
    os.makedirs(_temp_images, exist_ok=True)
except Exception:
    pass

print("[\033[92mWAS LMStudio Easy-Query\033[0m] has loaded {} nodes.".format(len(NODE_CLASS_MAPPINGS)))
print("Nodes: " + ", ".join([f"\033[1m{node_name}\033[0m" for node_name in NODE_DISPLAY_NAME_MAPPINGS.values()]))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
