#!/bin/sh

install() {
    package=$2
    version=$3
    [ -z ${package} ] && package=$1
    python -c "import ${package}"
    if [ $? -ne 0 ]
    then
        pip install $1==${version}
    fi
}

install pre-commit pre_commit 2.19.0

Config=.pre-commit-config-internal.yaml

py_files=$(find . -path "./atorch/protos" -prune -o -name "*.py" -print0 | tr '\0' ' ')
pre-commit run -v --files ${py_files} -c ${Config}

STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    echo "============================== Hello Atorch ================================="
    echo "|                                                                            |"
    echo "| Please check above error message.                                          |"
    echo "| You can run sh tools/scripts/pre-commit.sh locally                         |"
    echo "|                                                                            |"
    echo "============================== Hello Atorch ================================="
    exit ${STATUS}
fi