
set -e
echo "here is me:$ACI_COMMIT_REF_NAME"
export
# remove aliyun mirror
echo "
[global]
index-url = https://pypi.antfin-inc.com/simple/
" > ~/.pip/pip.conf 
pip install -i https://pypi.antfin-inc.com/simple wheel twine==1.12.1

# ACI_VAR_PYPI_PASS 是加密变量, 配置在：https://aci.alipay.com/project/13301290/setting?tab=variable
echo "[easy_install]
index-url=https://artifacts.antgroup-inc.cn/simple/

" > ~/.pydistutils.cfg

echo "[distutils]
index-servers =
    pypiantfin

[pypiantfin]
repository: https://artifacts.antgroup-inc.cn/simple/
username: $ACI_VAR_PYPI_USER
password: $ACI_VAR_PYPI_PASS

" > ~/.pypirc
case $ACI_COMMIT_REF_NAME in
  release_*)
    #release_ len: 8 打上tag形如release_1.0.1 那么将会产生 atorch_1.0.1版本上传
    VERSION=${ACI_COMMIT_REF_NAME:8}
    ;;
  *) 
  echo -n "invalid env:$ACI_COMMIT_REF_NAME"
  exit 1
  ;;
esac

if [ ! -z $VERSION ];then
    python dev/scripts/render_setup.py --version $VERSION
    python setup.py bdist_wheel
    python setup.py sdist
    ls dist
    twine upload -r pypiantfin "dist/atorch-${VERSION}-py3-none-any.whl" # python3
    twine upload -r pypiantfin "dist/atorch-${VERSION}.tar.gz"
fi
