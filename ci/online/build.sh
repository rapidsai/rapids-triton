#!/bin/bash

set -e

apt-get update

apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce-cli 

REPODIR=$(cd $(dirname $0)/../../; pwd)

EXAMPLE_TAG=rapids_triton_identity \
  TEST_TAG=rapids_triton_identity_test \
  $REPODIR/build.sh
if [ -z $CUDA_VISIBLE_DEVICES ]
then
  docker run --gpus all --rm rapids_triton_identity_test
else
  docker run --gpus $CUDA_VISIBLE_DEVICES --rm rapids_triton_identity_test
fi
EXAMPLE_TAG=rapids_triton_identity:cpu \
  TEST_TAG=rapids_triton_identity_test:cpu \
  $REPODIR/build.sh --cpu-only
docker run -v "${REPODIR}/qa/logs:/qa/logs" --gpus all --rm rapids_triton_identity_test:cpu
