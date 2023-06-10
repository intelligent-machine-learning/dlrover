#!/bin/sh
TYPE=$1
[ -z ${TYPE} ] && TYPE=release

install() {
    package=$2
    [ -z ${package} ] && package=$1
    python -c "import ${package}"
    if [ $? -ne 0 ]
    then
        pip install $1
    fi
}

install_fix() {
    pip install $1
}


install pre-commit pre_commit

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

pip list