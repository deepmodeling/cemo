#!/usr/bin/zsh
# start tensorboard on kvm

dir=$1; shift

if [ -z $dir ]; then
    echo "Usage: $0 <logdir>"
    exit 1
fi

port=40000
host=172.16.1.223
tensorboard --port $port --host $host --logdir $dir
