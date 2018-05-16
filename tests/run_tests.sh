#!/bin/bash


for testfile in `ls tests/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile
done
