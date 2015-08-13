#!/bin/bash

cd ~/Projects/sampyl_docs/sampyl
cp -r ../../sampyl/docs/* .
git commit -a -m "Updating documentation."
git push