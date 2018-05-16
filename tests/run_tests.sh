#!/bin/bash


for testfile in `ls tests/*.py`
do
	python -m pytest $testfile
done
