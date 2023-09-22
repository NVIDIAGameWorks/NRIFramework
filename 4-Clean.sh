#!/bin/sh

rm -rf "build"

rm -rf "_Build"
rm -rf "_Compiler"
rm -rf "_Shaders"
rm -rf "_NRI_SDK"

cd "External/NRI"
source "4-Clean.sh"
cd "../.."
