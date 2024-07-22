for i in $(seq 1 30);
do
    qsub -N burgers_generator_N8192_nu001_samples40_batch${i} -v BATCH=${i} burgers_generator_N8192_nu001_samples40.sh
done