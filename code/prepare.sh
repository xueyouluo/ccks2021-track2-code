#!/usr/bin/env bash

mkdir -p ../user_data/tcdata
mkdir -p ../user_data/texts

cp -r ../tcdata ../user_data

echo "Data preprocess"
python create_raw_text.py