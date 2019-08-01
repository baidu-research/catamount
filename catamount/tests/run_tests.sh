#!/bin/bash


echo -e "------------ Running ops tests ---------------\n\n"
for testfile in `ls catamount/tests/ops/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile -p no:warnings
done

echo -e "\n\n------------ Running API tests ---------------\n\n"
for testfile in `ls catamount/tests/api/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile -p no:warnings
done

echo -e "\n\n------------ Running full tests ---------------\n\n"
for testfile in `ls catamount/tests/full/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile -p no:warnings
done
