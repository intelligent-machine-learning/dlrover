#!/bin/sh
TYPE=$1
[ -z ${TYPE} ] && TYPE=release

pip install pre-commit==2.21.0

if [ ${TYPE} != "precommit" ]
then

    sh build.sh ${TYPE}
    pushd dist
    file=`ls -al | grep ".whl" | awk '{print $NF}'`
    pip install "${file}[tfplus]" --upgrade
    popd

    \rm -rf build dist
    pip install pytest -I
    pip install pytest-xdist -I
    pip install coverage -I
    install sklearn
    # install_fix keras==2.0.6
    pushd ../../third_party
    pip install --force-reinstall -U tensorflow
    popd
fi
