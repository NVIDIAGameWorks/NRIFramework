@echo off

if exist "build" rd /q /s "build"

if exist "_Build" rd /q /s "_Build"
if exist "_Compiler" rd /q /s "_Compiler"
if exist "_Shaders" rd /q /s "_Shaders"
if exist "External/Assimp" rd /q /s "External/Assimp"
if exist "External/Detex" rd /q /s "External/Detex"
if exist "External/Glfw" rd /q /s "External/Glfw"
if exist "External/ImGui" rd /q /s "External/ImGui"

cd "External/NRI"
call "4-Clean.bat"
cd "../.."
