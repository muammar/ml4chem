Building Documentation
=======

Documentation is a very important part of any project, and in ML4Chem special
attention is given to provide a clear documentation.

To locally build the docs you need to make sure all dependencies are correctly
installed::

   cd /path/ml4chem
   pip install -r requirements.txt
   cd /path/ml4chem/docs
   pip install -r requirements.txt

Also, you have to install `m2r` from the following git repository::

   pip install --upgrade 'git+https://github.com/crossnox/m2r@dev#egg=m2r'

Finally, execute the `makedocs.sh` script::

   sh makedocs.sh

This will automatically perform the following commands for you::

    cd source
    sphinx-apidoc -fo . ../../
    cd ..
    make html

When the Makefile is finished, you can check the documentation in the
`build/html` folder.
