#!/bin/bash

# Set up virtual environment
if [ ! -e "catamount_venv" ]
then
	echo "==== Creating virtual environment ===="
	python -m venv catamount_venv
	echo ""
fi

echo "==== Sourcing virtual environment ===="
source catamount_venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.6/site-packages/
echo ""

echo "==== Installing dependencies ===="
pip install -r requirements.txt
echo ""

echo "==== Installing Catamount ===="
python setup.py install
echo ""


# Run tests
# Use CUDA_VISIBLE_DEVICES="" in case system has access to GPUs,
# but we don't want to use them
for depth in 18 34 50 101 152
do
	echo "==== Running Image ResNet$depth. Output to output_image_$depth.txt ===="
	CUDA_VISIBLE_DEVICES="" python catamount/tests/full/tf_image_resnet.py --depth $depth >& output_image_$depth.txt
	echo ""
done

for domain in charlm nmt wordlm
do
	echo "==== Running Language $domain. Output to output_$domain.txt ===="
	CUDA_VISIBLE_DEVICES="" python catamount/tests/full/tf_language_models.py --domain $domain >& output_$domain.txt
	echo ""
done

echo "==== Running Speech Attention. Output to output_speech.txt ===="
CUDA_VISIBLE_DEVICES="" python catamount/tests/full/tf_speech_attention.py >& output_speech.txt
echo ""


# Gather results


