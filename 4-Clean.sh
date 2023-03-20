#!/bin/sh

rm -rf "build"

rm -rf "_Build"
rm -rf "_Compiler"
rm -rf "_Shaders"
rm -rf "_NRI_SDK"
rm -rf "External/Assimp"
rm -rf "External/Detex"
rm -rf "External/Glfw"
rm -rf "External/ImGui"

cd "External/NRI"
source "4-Clean.sh"
cd "../.."
