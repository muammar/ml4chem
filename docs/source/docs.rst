Build the docs
=======

To locally build the docs you need to perform the following::

    cd mlchem/docs/source
    sphinx-apidoc -fo . ../../
    cd ..
    make html
