#!/bin/bash

rm -f .bazelrc
if python -c "import tensorflow" &> /dev/null; then
    echo 'using installed tensorflow'
else
    pip install tensorflow==2.12.0
fi

python config_helper.py