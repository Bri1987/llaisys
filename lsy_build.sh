#! /bin/bash
xmake clean --all --root
xmake f --nv-gpu=y -cv --root
xmake --root
xmake install --root
pip install ./python/