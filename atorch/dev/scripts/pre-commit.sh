#!/bin/sh

pip install pre-commit==2.21.0
git config --global --add safe.directory '*'

Config=.pre-commit-config.yaml

py_files=$(find . -path "atorch" -prune -o -name "*.py" -print0 | tr '\0' ' ')
pre-commit run -v --files ${py_files} -c ${Config}

STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    echo "============================== Hello Atorch ================================="
    echo "|                                                                            |"
    echo "| Please check above error message.                                          |"
    echo "| You can run sh dev/scripts/pre-commit.sh locally                           |"
    echo "|                                                                            |"
    echo "============================== Hello Atorch ================================="
    exit ${STATUS}
fi