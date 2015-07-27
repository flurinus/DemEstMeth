#!/bin/bash

if [ -n "$1" ] 
then
n=$1
echo "Parsing table and creating Circos input files"
cat input/$n | bin/parse-table -conf etc/parse-table_v3.conf | bin/make-conf -dir data/
echo "Drawing Circos image"
bin/circos -conf circos.conf
else
echo "Error"
fi