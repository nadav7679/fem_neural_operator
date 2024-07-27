for i in $(seq 1 30);
do
    qsub -N burgers_generator_N8192_nu001_samples40_batch${i} -v BATCH=${i} burgers_generator_N8192_nu001_samples40.sh
done

for i in $(seq 1 2);
do
    qsub -N KS_generator_N4096_nu0029_T_01_samples600_batch${i} -v BATCH=${i} KS_generator_N4096_nu0029_T01_samples600.sh
done