#!/usr/bin/env sh

sudo apt update && sudo apt install -y xvfb
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99