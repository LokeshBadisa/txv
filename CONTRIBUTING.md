# Contributor Guidelines

This project uses sphinx as documentation tool. The documentation is written in reStructuredText format.
Please look docstring style being used(called numpy/google style). It is recommended to follow the same style and this style is made possible through napoleon extension of sphinx.

### How to build the documentation
1. Install required packages.
```bash
pip install -r dev-requirements.txt
```

2. Execute the following command to build the documentation. Save it as a bash alias to save time.
```bash
sphinx-apidoc -o docs/source txv/ --no-toc && cd docs && make clean html && make html && cd ..
```

Any contribution to either documentation or code is welcome. 