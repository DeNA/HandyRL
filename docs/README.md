Checkout `gh-pages`.
```
git checkout master
git checkout gh-pages
git checkout -b feature/update-gh-pages
git merge master
```

Install sphinx and the related libraries.
```
pip install -r requirements.txt
```

```
sphinx-quickstart docs
```

Generate rst files for new docs.
```
sphinx-apidoc -f -e -o ./docs/source ./handyrl
```

Build docs.
```
cd docs
make html
```

Copy built docs to `docs` folder.
```
cp docs/build/* docs
```