#!/bin/bash
set -e
set -u

SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --rm --privileged -v /dev:/dev -v ${SCRIPTROOT}/..:/home/root/rl_ws --network=host --gpus=all --name=walk-these-way_container walk-these-way /home/root/rl_ws/setup.sh
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --rm --privileged -v /dev:/dev -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ${SCRIPTROOT}/..:/home/root/rl_ws -v /home/fzq/.Xauthority:/root/.Xauthority --network=host --gpus=all --name=walk-these-way_container walk-these-way /home/root/rl_ws/setup.sh
fi
