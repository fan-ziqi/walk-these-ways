#!/bin/bash
set -e
set -u
# SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
# cd "${SCRIPTROOT}/.."

docker build --network host -t walk-these-way -f Dockerfile .
