set -e

rm -rf build
# NOTE: The following commands requires that the current working directory is
# the source tree root.
make -f dlrover/Makefile

# Create elasticdl_preprocessing package
echo "Building the wheel for elasticdl_preprocessing."
rm -rf ./build/lib
python setup.py --quiet bdist_wheel --dist-dir ./build
