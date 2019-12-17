#!/bin/bash -e
pip install six numpy wheel setuptools mock 'future>=0.17.1'
pip install keras_applications --no-deps
pip install keras_preprocessing --no-deps

bazel build -c opt --copt=-g --config=cuda //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install -U /tmp/tensorflow_pkg/tensorflow-2.0.0b1-cp36-cp36m-linux_x86_64.whl

