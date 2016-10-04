#!/bin/bash
rm  -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release
if [[ -f shape_predictor_68_face_landmarks.dat ]]; then
	echo "já tem arquivo"
else
	wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	bunzip2 shape_predictor_68_face_landmarks.dat.bz2
fi
