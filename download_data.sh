#!/bin/bash
NAUS_data_link=https://drive.google.com/drive/folders/1Gc2vRMscoP111xMhAJozfCdcr5cuPJQQ
gdown --folder $NAUS_data_link
unzip training_data/gigaword_10.zip
unzip training_data/gigaword_13.zip
unzip training_data/gigaword_8.zip
unzip training_data/gigaword_ref.zip
rm -rf training_data/gigaword_10.zip
rm -rf training_data/gigaword_8.zip
rm -rf training_data/gigaword_13.zip
rm -rf training_data/gigaword_ref.zip
