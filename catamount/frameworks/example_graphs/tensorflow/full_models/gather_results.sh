#!/bin/bash

outdir="ppopp_2019_tests"

ls $outdir

echo "=============== Character Language Model ==============="
grep -A50000 "======= Algorithmic graph-level analytics: =======" $outdir/output_charlm.txt | grep -v "====="

echo -e "\n\n"

echo "=============== Word Language Model ==============="
grep -A50000 "======= Algorithmic graph-level analytics: =======" $outdir/output_wordlm.txt | grep -v "====="

echo -e "\n\n"

echo "=============== Machine Translation Model ==============="
grep -A50000 "======= Algorithmic graph-level analytics: =======" $outdir/output_nmt.txt | grep -v "====="

echo -e "\n\n"

echo "=============== Image Classification Models ==============="
for depth in 18 34 50 101 152
do
	echo "--------------- ResNet-$depth ---------------"
	grep -A50000 "======= Algorithmic graph-level analytics: =======" $outdir/output_image_$depth.txt | grep -v "====="
	echo ""
done

echo -e "\n\n"

echo "=============== Speech Recognition Model ==============="
grep -A50000 "======= Algorithmic graph-level analytics: =======" $outdir/output_speech.txt | grep -v "====="
