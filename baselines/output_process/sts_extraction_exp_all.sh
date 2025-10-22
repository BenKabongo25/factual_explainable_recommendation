#!/bin/bash

JOB_SCRIPT="baselines/output_process/sts_extraction_exp.sh"

LOG_DIR="logs/sts_extraction"
mkdir -p "${LOG_DIR}"

datasets=(Toys Clothes Beauty Sports Cell)

echo "Submitting jobs via: ${JOB_SCRIPT}"

job_count=0

methods=(Att2Seq_explanation NRT_explanation CER_explanation PETER_explanation PEPLER_explanation XRec)
for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    job_count=$((job_count + 1))
    job_name="${method}_${dataset}"
    out_path="${LOG_DIR}/${job_name}.out"
    err_path="${LOG_DIR}/${job_name}.err"
      
    jid=$(sbatch --parsable \
        -J "${job_name}" \
        -o "${out_path}" \
        -e "${err_path}" \
        "${JOB_SCRIPT}" "${dataset}" "${method}" prediction)
      
    echo "[$job_count] Submitted: dataset=${dataset} method=${method} jobid=${jid}"
    echo "     logs: ${out_path} | ${err_path}"
  done
done

echo ""
echo "All $job_count jobs submitted!"