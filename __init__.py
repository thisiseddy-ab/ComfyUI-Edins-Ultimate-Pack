import folder_paths
import os
import sys
import importlib

# Get paths
comfy_path = os.path.dirname(folder_paths.__file__)
eup_repo_path = os.path.dirname(__file__)
EUP_directory_path = os.path.join(eup_repo_path, "EUP")

# Ensure the parent directory is in sys.path
parent_dir = os.path.dirname(EUP_directory_path)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Debugging: Verify paths
print("EUP directory exists:", os.path.exists(EUP_directory_path))
print("sys.path includes parent dir:", parent_dir in sys.path)

_modules_to_import_ = [
    "basic_utils"
    ]
_nodes_to_import_ = [

    "latent",
    "upscaler",
    ]

NODE_CLASS_MAPPINGS = {

}
__all__ = ["NODE_CLASS_MAPPINGS"]

for module_name in _modules_to_import_:
    try:
        module = importlib.import_module(f"EUP.utils.{module_name}")
        globals()[module_name] = module
        __all__.append(module_name)
    except ModuleNotFoundError as e:
        print(f"Error importing {module_name}: {e}")

for nodes_name in _nodes_to_import_:
    try:
        nodes_module = importlib.import_module(f"EUP.nodes.{nodes_name}")
        globals()[nodes_name] = nodes_module
        __all__.append(nodes_name)

        # Check before merging to avoid errors
        if hasattr(nodes_module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **nodes_module.NODE_CLASS_MAPPINGS}
        else:
            print(f"Warning: {nodes_name} does not have NODE_CLASS_MAPPINGS, skipping merge.")
    except ModuleNotFoundError as e:
        print(f"Error importing {nodes_name}: {e}")