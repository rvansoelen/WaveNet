#!/bin/bash

for filename in paramFiles/*.json; do
	name=${filename##*/}
        base=${name%.json}
	echo Running $base
	mkdir -p output/$base
	python3 ../train.py --wavenet_params $filename --data_dir ../../WaveNet/data/wavenet_corpus --logdir output/$base
done
