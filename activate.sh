# Run
#
#  $ source activate.sh
#
# to activate the TensorFlow/TensorSwift environment maintained by the LM team.
# This script also sets up git hooks so that flake8 tests are performed
# before you can check in python files

if [[ "$(basename -- "$0")" == "activate.sh" ]]; then
    echo "Don't run $0, source it" >&2
    exit 1
fi

# Source the virtual environment
if [[ ! -z "$LOCAL_VIRTUAL_ENV" ]]; then
    # To use a local virtual environment other than the default for this
    # project, specify its path in the environment variable LOCAL_VIRTUAL_ENV
    echo "WARN:"
    echo "WARN: Using specified virtual env: $LOCAL_VIRTUAL_ENV"
    echo "WARN:"
    source $LOCAL_VIRTUAL_ENV/bin/activate_and_save
else
    if [ "`hostname | sed 's/.svail.baidu.com//' | grep -o svail`" == "svail" ]
    then
        source /tools/lm-venv/py3.6.1-tf-1.5.0rc0-svail/bin/activate_and_save
    elif [ "`hostname | sed 's/.svail.baidu.com//' | grep -o asimov`" == "asimov" ]
    then
        source /tools/lm-venv/py3.6.5-tf-1.7.0-cuda-9.2-asimov/bin/activate_and_save
    else
        echo "ERROR: Unknown hostname (`hostname`). Unable to find virtual environment"
    fi
fi
