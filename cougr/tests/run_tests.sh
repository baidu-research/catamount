#!/bin/bash


echo -e "------------ Running ops tests ---------------\n\n"
for testfile in `ls cougr/tests/ops/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile
done

echo -e "\n\n------------ Running API tests ---------------\n\n"
for testfile in `ls cougr/tests/api/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile
done

echo -e "\n\n------------ Running full tests ---------------\n\n"
for testfile in `ls cougr/tests/full/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile
done
