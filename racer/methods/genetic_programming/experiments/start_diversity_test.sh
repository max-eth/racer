BASE_COMMAND='python -m racer.methods.genetic_programming.genetic_programming -F'
WITH_PREFIX='with car_racing_env.headless=True parallel=False show_best=False'

run() {
    number=$1
    shift
    name=$1
    shift
    for i in `seq $number`; do
      echo bsub -J $name-$i -n 1 -W 5:30 -R "rusage[mem=3000]" $BASE_COMMAND logs/$name-$i $WITH_PREFIX $@ 
    done
}

for n_elitism in "0 1 2 3 4 8 16 32"
do
  for tournament_size in "2 4 8 16 32"
     do
       run 4 n_elitism$n_elitism-t_size$tournament_size n_elitism=$n_elitism selector_params.tournamen_size=$tournament_size
     done
done

