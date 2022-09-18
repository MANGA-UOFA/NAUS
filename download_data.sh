#!/bin/bash
NAUS_data_link=https://drive.google.com/drive/folders/1XKN6oFy2-C6ChkfjUVIJHXFCqTVF9vjo?usp
gdown --folder $NAUS_data_link
unzip training_data/gigaword_10.zip
unzip training_data/gigaword_13.zip
unzip training_data/gigaword_8.zip
unzip training_data/gigaword_ref.zip
rm -rf training_data/gigaword_10.zip
rm -rf training_data/gigaword_8.zip
rm -rf training_data/gigaword_13.zip
rm -rf training_data/gigaword_ref.zip
