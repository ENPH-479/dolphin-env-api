#!/usr/bin/env python

"""
setup.py file for PyDolphinMemoryEngine
"""

from distutils.core import setup, Extension


PyDolphinMemoryEngine = Extension('_PyDolphinMemoryEngine',
                           sources=['PyDolphinMemoryEngine_wrap.cxx', 'Dolphin-memory-engine/Source/MemoryWatch/MemoryWatch.cpp', 'Dolphin-memory-engine/Source/MemoryScanner/MemoryScanner.cpp', 'Dolphin-memory-engine/Source/DolphinProcess/DolphinAccessor.cpp', 'Dolphin-memory-engine/Source/DolphinProcess/Linux/LinuxDolphinProcess.cpp', 'Dolphin-memory-engine/Source/Common/MemoryCommon.cpp'],
                           )

setup (name = 'PyDolphinMemoryEngine',
       version = '0.1',
       author      = "0x4249",
       description = """Python module of Aldelaro5's Dolphin Memory Engine""",
       ext_modules = [PyDolphinMemoryEngine],
       py_modules = ["PyDolphinMemoryEngine"],
       )

