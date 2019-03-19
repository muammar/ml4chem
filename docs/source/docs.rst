Building Documentation
=======

Documentation is a very important part of any project, and in MLChem special
attention is given to provide a clear documentation.

To locally build the docs you need to perform the following::

    cd mlchem/docs/source
    sphinx-apidoc -fo . ../../
    cd ..
    make html
