#!/usr/bin/env sh

rm -rf build
cd source
sphinx-apidoc -fo . ../../
cd ..
make html
