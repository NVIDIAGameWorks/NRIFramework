@echo off

if exist "build" rd /q /s "build"

if exist "_Bin" rd /q /s "_Bin"
if exist "_Build" rd /q /s "_Build"
if exist "_Shaders" rd /q /s "_Shaders"

cd "External/NRI"
call "4-Clean.bat"
cd "../.."
