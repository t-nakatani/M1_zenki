#!/bin/bash

g++ routing_simulation.cpp -o sample.out
g++ report14_BGA22078.cpp -o myQnet.out

for i in `seq 4`
do
  
  echo "sample_output($i): `./sample.out $i`"

  echo "myQnet_output($i): `./myQnet.out $i`\n"
  
done