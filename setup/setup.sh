#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating and navigating to project directory..."
git clone https://github.com/C0nsumption/Consume-CogVLM2.git
cd Consume-CogVLM2

echo "Setting up a virtual environment..."
python -m venv venv
source venv/bin/activate


echo "Starting Git LFS..."
git lfs install

echo "Cloning the model repository..."
echo "TAKES A WHILE IF SLOW INTERNET..."
git clone https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4

echo "Installing torch 2.3.0..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tests..."
python test/test.py
