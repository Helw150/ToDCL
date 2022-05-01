#!/bin/bash
# SGD dataset
git clone --depth 1 https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git &
git clone --depth 1 https://github.com/google-research-datasets/Taskmaster.git &
git clone --depth 1 https://github.com/budzianowski/multiwoz.git &

# MWOZ
pip install absl-py

wait

cd multiwoz/data
unzip MultiWOZ_2.1
cd MultiWOZ_2.2
python convert_to_multiwoz_format.py --multiwoz21_data_dir=../MultiWOZ_2.1 --output_file=data.json
cd ../../../../
