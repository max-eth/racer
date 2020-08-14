BASE_COMMAND='python -m racer.methods.genetic_programming.genetic_programming -F'
WITH_PREFIX='with car_racing_env.headless=True parallel=False show_best=False use_schedule=False'

run() {
    number=$1
    shift
    name=$1
    shift
    for i in `seq $number`; do
      echo bsub -J $name-$i -n 1 -W 5:30 -R "rusage[mem=3000]" $BASE_COMMAND logs/$name-$i $WITH_PREFIX $@ 
    done
}

for gen_gauss in "True" "False"  # whether to use 1,2,3 or generate constant
do
  for min_max in "2 4" "6 8" "8 12" # heights
     do
       set -- $min_max  # now min height in $1 and max height in $2
       run 4 gen_gauss$gen_gauss-height_$1_$2 gen_gauss=$gen_gauss min_height=$1 max_height=$2
     done
done
