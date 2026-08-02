#!/bin/bash

set -e

BUILD_DIR="build"
EXECUTABLE="JsonManager"

if [ ! -d "$BUILD_DIR" ]; then
	echo "Creating build directory"
	mkdir "$BUILD_DIR"
fi

#Configure Cmake
cmake -S . -B "$BUILD_DIR"

#Build project
cmake --build "$BUILD_DIR"

#Run executable
if [ -f "$BUILD_DIR/$EXECUTABLE" ]; then
	"$BUILD_DIR/$EXECUTABLE"
else
	echo "Executable not found!"
	exit 1
fi
