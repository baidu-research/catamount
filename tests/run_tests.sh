#!/bin/bash


echo "------------ Running ops tests ---------------"
for testfile in `ls tests/ops/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile
done

echo "------------ Running full tests ---------------"
for testfile in `ls tests/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile
done
