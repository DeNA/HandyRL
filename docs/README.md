TODO: CI/CD when develop branch is merged into master branch.

Checkout `gh-pages`.
```
git checkout master
git pull
git checkout -b feature/update-gh-pages
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
cp -a build/html/. .
rm -r build/
```

Push the new docs.
```
# git add ...
# git commit ...
# git push origin feature/update-gh-pages
```

Source files: PR to master
Build files: PR to `gh-pages`
