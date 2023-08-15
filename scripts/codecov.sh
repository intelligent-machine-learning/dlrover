echo $CODECOV_TOKEN
curl -Os https://uploader.codecov.io/latest/linux/codecov
chmod +x codecov
./codecov