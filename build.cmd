@echo off
setlocal
set "CV40_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%CV40_VSWHERE%" (
  echo Install Visual Studio with Desktop development with C++ first.
  exit /b 1
)
for /f "usebackq tokens=*" %%I in (`"%CV40_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "CV40_VS=%%I"
if not defined CV40_VS (
  echo No MSVC x64 compiler found.
  exit /b 1
)
call "%CV40_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
where cmake >nul 2>&1
if errorlevel 1 (
  echo CMake 3.24 or newer must be on PATH.
  exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 (
  echo Ninja must be on PATH.
  exit /b 1
)
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" --parallel 6
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build" --output-on-failure
exit /b %errorlevel%
