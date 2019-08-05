#!/bin/bash

testsfailed="0"
failedtests=()
testspassed="0"

echo -e "------------ Running ops tests ---------------\n\n"
for testfile in `ls catamount/tests/ops/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile -p no:warnings
	if [ "$?" != "0" ]
	then
		failedtests[$testsfailed]="$testfile"
		testsfailed=$(($testsfailed + 1))
	else
		testspassed=$(($testspassed + 1))
	fi
done

echo -e "\n\n------------ Running API tests ---------------\n\n"
for testfile in `ls catamount/tests/api/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile -p no:warnings
	if [ "$?" != "0" ]
	then
		failedtests[$testsfailed]="$testfile"
		testsfailed=$(($testsfailed + 1))
	else
		testspassed=$(($testspassed + 1))
	fi
done

echo -e "\n\n------------ Running full tests ---------------\n\n"
for testfile in `ls catamount/tests/full/*.py`
do
	echo Running test: $testfile
	python -m pytest $testfile -p no:warnings
	if [ "$?" != "0" ]
	then
		failedtests[$testsfailed]="$testfile"
		testsfailed=$(($testsfailed + 1))
	else
		testspassed=$(($testspassed + 1))
	fi
done

echo -e "\n\n------------ Tests summary ---------------\n\n"
totaltests=$(($testsfailed + $testspassed))
echo "Tests passed: $testspassed"
echo "Tests failed: $testsfailed"
for i in "${failedtests[@]}"
do
	echo "        $i"
done
