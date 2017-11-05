Ingredients for generating a Python wrapper for the Dolphin-memory-engine project: 

https://github.com/aldelaro5/Dolphin-memory-engine

using SWIG. Run the following commands on Linux to generate the Python wrapper: 

swig -c++ -python -py3 PyDolphinMemoryEngine

python setup.py build_ext --inplace



(Ensure you are using Python 3!)
