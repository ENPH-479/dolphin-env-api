#!/bin/bash

CFG_DIR=".dolphin-emu"

echo "=================================="
echo "Automated Dolphin Pipe Input Setup"
echo "=================================="
echo

if [ -d ~/"$CFG_DIR" ]; then
  echo "Dolphin user config folder found."
else
  echo "Dolphin config folder not found."
  mkdir -p ~/"$CFG_DIR"
  echo "Creating Dolphin config folder..."
fi

echo "=================================="
echo "Creating pipe input..."
mkdir -p ~/"$CFG_DIR"/Pipes
mkfifo ~/"$CFG_DIR"/Pipes/pipe

echo "=================================="
echo "Copying controller profile..."
mkdir -p ~/"$CFG_DIR"/Config/Profiles/GCPad
cp ./bin/bot.ini ~/"$CFG_DIR"/Config/Profiles/GCPad

echo "=================================="
read -p "Do you want to set Player 1 to be bot-controlled now?" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Setting Controller 1 to bot-controlled profile..."
  echo "[GCPad1]" > ~/"$CFG_DIR"/Config/GCPadNew.ini
  tail --lines=+2 ./bin/bot.ini  >> ~/"$CFG_DIR"/Config/GCPadNew.ini
fi

echo "=================================="
echo "Done."
echo

