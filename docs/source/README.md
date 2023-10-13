To populate the table of contents automatically, open a console and navigate to the main package directory:
```commandline
sphinx-apidoc -f -o docs/source mirp/    
```

By default, only part of the API is exposed to the user.

To build html files, open a console, navigate to `./docs` directory:
```commandline
.\make html
```
