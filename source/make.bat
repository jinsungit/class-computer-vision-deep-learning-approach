@echo off
REM Minimal make.bat for Sphinx on Windows
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="html" (
    %SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html
) else if "%1"=="clean" (
    if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
) else (
    echo Usage: make.bat html ^| clean
)
