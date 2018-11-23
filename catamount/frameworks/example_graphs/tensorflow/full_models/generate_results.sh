#!/bin/bash

# Check for correct Python version
pyversion=`python --version 2>&1 | grep -o "[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"`
major=`echo $pyversion | sed 's/\..*//'`
minor=`echo $pyversion | sed 's/[0-9]*\.\([0-9]*\)\..*/\1/'`
if [ "$major" != "3" ] || [ "$minor" != "6" ]
then
	USEPYTHON=`which python3.6`
	if [ ! -e "$USEPYTHON" ]
	then
		echo "ERROR: Python version $pyversion not supported. Please use Python 3.6"
		exit
	fi
else
	USEPYTHON=`which python`
fi

# Set up virtual environment
if [ ! -e "catamount_venv" ]
then
	echo "==== Creating virtual environment ===="
	$USEPYTHON -m venv catamount_venv
	echo ""
fi

echo "==== Sourcing virtual environment ===="
activatescript="catamount_venv/bin/activate"
if [ ! -e "$activatescript" ]
then
	echo "ERROR: Failed to create virtualenv, catamount_venv. Unable to find $activatescript"
	exit
fi
source catamount_venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.6/site-packages/
echo ""

echo "==== Installing dependencies ===="
pip install -r requirements.txt
echo ""

echo "==== Installing Catamount ===="
$USEPYTHON setup.py install
echo ""

outdir="ppopp_2019_tests"
if [ ! -e "$outdir" ]
then
	echo "==== Creating output directory ===="
	mkdir -p $outdir
	echo ""
fi

# Run tests
# Use CUDA_VISIBLE_DEVICES="" in case system has access to GPUs,
# but we don't want to use them
for depth in 18 34 50 101 152
do
	outfile="$outdir/output_image_$depth.txt"
	echo "==== Running Image ResNet$depth. Output to $outfile ===="
	CUDA_VISIBLE_DEVICES="" $USEPYTHON catamount/tests/full/tf_image_resnet.py --depth $depth >& $outfile
	echo ""
done

for domain in charlm nmt wordlm
do
	outfile="$outdir/output_$domain.txt"
	echo "==== Running Language $domain. Output to $outfile ===="
	CUDA_VISIBLE_DEVICES="" $USEPYTHON catamount/tests/full/tf_language_models.py --domain $domain >& $outfile
	echo ""
done

outfile="$outdir/output_speech.txt"
echo "==== Running Speech Attention. Output to $outfile ===="
CUDA_VISIBLE_DEVICES="" $USEPYTHON catamount/tests/full/tf_speech_attention.py >& $outfile
echo ""


# Gather results
bash catamount/frameworks/example_graphs/tensorflow/full_models/gather_results.sh
