#!/usr/bin/env bash

input_dir=$1
no_of_files=$2

ls ${input_dir} | shuf -n ${no_of_files} | xargs -I {} mv ${input_dir} ./${input_dir}_${no_of_files}
