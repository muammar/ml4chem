#!/usr/bin/env sh

cd source
sphinx-apidoc -fo . ../../
cd ..
make html
