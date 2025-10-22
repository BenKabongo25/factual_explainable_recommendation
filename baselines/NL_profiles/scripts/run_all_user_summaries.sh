#!/bin/bash


JOB_SCRIPT="baselines/NL_profiles/scripts/generate_user_summaries.sh"
LOG_DIR="logs/NL_profiles_user"
mkdir -p "${LOG_DIR}"

datasets=(Toys Clothes Beauty Sports Cell)

echo "Submitting jobs via: ${JOB_SCRIPT}"

job_count=0
method=user
for dataset in "${datasets[@]}"; do
    job_count=$((job_count + 1))
    job_name="${method}_${dataset}"
    out_path="${LOG_DIR}/${job_name}.out"
    err_path="${LOG_DIR}/${job_name}.err"
      
    jid=$(sbatch --parsable \
        -J "${job_name}" \
        -o "${out_path}" \
        -e "${err_path}" \
        "${JOB_SCRIPT}" "${dataset}")
      
    echo "[$job_count] Submitted: dataset=${dataset} method=${method} jobid=${jid}"
    echo "     logs: ${out_path} | ${err_path}"
done