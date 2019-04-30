Building Documentation
=======

Documentation is a very important part of any project, and in ML4Chem special
attention is given to provide a clear documentation.

To locally build the docs you need to execute the `makedocs.sh` script::

   sh makedocs.sh

This will automatically perform the following commands for you::

    cd source
    sphinx-apidoc -fo . ../../
    cd ..
    make html

When the Makefile is finished, you can check the documentation in the
`build/html` folder.
