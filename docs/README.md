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
cp -r docs/tutorial docs/documentation docs/faq docs/source
cd docs
make html
```
