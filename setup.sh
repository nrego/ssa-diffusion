#!/bin/bash

SSA_PYTHON=`which python2.7` 

echo "Using python: $SSA_PYTHON"
echo "(If this is not what you expect, try closing and reopening the terminal, and inspecting your 'PATH' enviornmental variable)"


PKGS=(numpy matplotlib yaml cython)

for pkg in ${PKGS[*]}; do

	$SSA_PYTHON -c "import $pkg" || echo "ERROR: Python package $pkg not found (Try downloading Anaconda or Canopy distribution"

done

echo "Looks like all Python packages are there..."
shebang="#!$SSA_PYTHON"

echo "Adding shebang line: '$shebang' to binaries..."

sed -i "1 s|.*|$shebang|" ./bin/ssa_*
