import lit.formats
import os

config.name = "SNN"
config.test_format = lit.formats.ShTest(not lit_config.useValgrind)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.snn_obj_root

config.substitutions.append(("%snn-opt", config.snn_opt_abs))
config.substitutions.append(("%FileCheck", config.filecheck_abs))
