#!/bin/bash
#install required package
apt-get install build-essential
apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
#install opencv
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
cd ../..
#install dlib
rm  -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release
