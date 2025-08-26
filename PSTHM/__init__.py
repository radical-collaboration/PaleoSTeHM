import pkgutil
import importlib

__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = importlib.import_module(f'.{module_name}', package=__name__)
    globals()[module_name] = _module