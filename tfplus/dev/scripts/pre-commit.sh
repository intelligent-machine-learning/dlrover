#!/bin/sh

DIR=`dirname $0`
sh ${DIR}/prepare.sh precommit

[ -e build ] && rm -rf build

Config=.pre-commit-config.yaml$1

echo "Precommit run without deploy folder"

pre-commit run -v --files $(find . -path ./deploy -prune -o -name "*.py" -print0 | tr '\0' ' ') -c ${Config}

STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    echo "============================== Hello  World ================================="
    echo "|                                                                            |"
    echo "| Please check above error message.                                          |"
    echo "| You can run sh dev/scripts/pre-commit.sh locally                               |"
    echo "|                                                                            |"
    echo "============================== Hello  World ================================="
    exit ${STATUS}
fi

echo "Precommit run in deploy"

pre-commit run -v --files $(find deploy -name "*.py" -print0 | tr '\0' ' ') -c ${Config}

STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    echo "============================== Hello  World ================================="
    echo "|                                                                            |"
    echo "| Please check above error message.                                          |"
    echo "| You can run sh scripts/pre-commit.sh locally                               |"
    echo "|                                                                            |"
    echo "============================== Hello  World ================================="
    exit ${STATUS}
fi
