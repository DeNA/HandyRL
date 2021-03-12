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

Modify and build docs.
```
cd docs
make html
```

Copy built docs to `docs` folder.
```
cd docs
cp -a build/html .
```

Push the new docs.
```
# git add ...
# git commit ...
# git push origin feature/update-gh-pages
```