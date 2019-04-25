Report for my bachelor project "Implementation of a deep learning library in Futhark"

## To compile to pdf run:

```
$ pdflatex -synctex=1 --shell-escape -interaction=nonstopmode master.tex
$ bibtex master.aux
$ pdflatex -synctex=1 --shell-escape -interaction=nonstopmode master.tex
```

Note that you need to have `Pygments` installed as `minted` depends on it.
Can be installed with `easy_install Pygments`