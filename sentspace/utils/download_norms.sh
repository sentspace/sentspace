#!/usr/bin/sh

set -ex

mkdir -p ~/.cache/sentspace

wget "https://www.dropbox.com/scl/fi/vphxhuwrv307hdrxgvicl/sentspace-norms.tar.gz?rlkey=s446xjk9u8k4xu2wt98cf8ndy&e=1&st=h696xwc2&dl=1" -O ~/.cache/sentspace/sentspace-norms.tgz

cd ~/.cache/sentspace/ && tar xzvf sentspace-norms.tgz
