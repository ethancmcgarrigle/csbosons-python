#!/bin/bash

# U=(0.0)
gamma=(-2.5 -2.25 -2.0 -1.75 -1.5 -1.25 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5) 
#lambda_psi=(50.0 10.0 1.0 0.75 0.5 0.25 0.1 0.05 0.01 0.005)
lambda_psi=(50.0) 

for gamma_ in ${gamma[@]}; do
  dir_name=gamma_${gamma_}
    if [ ! -d "./${dir_name}" ]
      then
      # if the directory does not exist, create it
      mkdir ${dir_name}
    fi
  cd ${dir_name}
  for lambda_psi_ in ${lambda_psi[@]}; do
    dir_name=lambda_psi_${lambda_psi_}
      if [ ! -d "./${dir_name}" ]
        then
        # if the directory does not exist, create it
        mkdir ${dir_name}
      fi
    cd ${dir_name}
    pwd
    # cp ../../*.py ./ 
    cp ../../submit.sh ./submit.sh
    cp ../../params.yml ./params.yml
    sed -i "s/__gamma__/${gamma_}/g" params.yml
    sed -i "s/__lambda_psi__/${lambda_psi_}/g" params.yml
    sed -i "s/__jobname__/${dir_name}/g" submit.sh
    # submit the job
    qsub submit.sh
    cd ..
  done 
  cd ..
done

