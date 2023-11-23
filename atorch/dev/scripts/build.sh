set -e
if [ -z "$1" ]; then
	version=0.1.0dev
else
	version=$1
fi
echo "Building ATorch version $version"
python dev/scripts/render_setup.py --version $version
python setup.py bdist_wheel
