#!/bin/bash
NAUS_data_link=https://drive.google.com/drive/folders/1XKN6oFy2-C6ChkfjUVIJHXFCqTVF9vjo?usp
gdown --folder $NAUS_data_link
unzip gigaword_10.zip
unzip gigaword_13.zip
unzip gigaword_8.zip
unzip gigaword_ref.zip
rm -rf gigaword_10.zip
rm -rf gigaword_8.zip
rm -rf gigaword_13.zip
rm -rf gigaword_ref.zip

mkdir model_weights
mv *.pt model_weights
