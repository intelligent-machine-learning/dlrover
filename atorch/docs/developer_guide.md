# Introduction to develop DlRover/atorch

The document describes how to make contribution to atorch in DLRover, see DlRover/docs/developer_guide.md firstly.

## Submit a PR

- Fork DLRover Repo to your owner namespace.
- `git clone git@github.com:intelligent-machine-learning/dlrover.git`
- `cd dlrover`
- `git remote rename origin upstream`
- `git remote add origin ${YOUR OWNER REPO}`
- `git checkout -b {DEV-BRANCH}`
- `git push -u origin {DEV-BRANCH}`

Then, you create a PR on your own github repo. If you has modified DLRover/atorch codes of the repo,
you need to execute `pre-commit` to check codestyle and unittest cases
by the following steps.

- ```docker run -v `pwd`:/dlrover -it easydl/atorch:aci /bin/bash```
- `cd /dlrover/atorch`
- `bash dev/scripts/build_proto.sh`
- `bash dev/scripts/pre-commit.sh`
- `exit`
- ```docker run -v `pwd`:/dlrover -it easydl/atorch:iml_pt210 /bin/bash```
- `pip install pytest dlrover[torch] fairscale==0.4.1 pandas Gpy `
- `pip install accelerate datasets==2.14.6 peft==0.4.0 scikit-learn pymoo==0.5.0`
- `echo -e 'import math\ninf = math.inf\nnan = math.nan\nstring_classes = (str, bytes)' > /opt/conda/lib/python3.8/site-packages/torch/_six.py`
- `PYTHONPATH=. pytest atorch/tests`
- `cd ..`
- `git config --global --add safe.directory /github/workspace`
- `git clean -xdf`

Otherwise，follow the testing steps in DlRover/docs/developer_guide.md.