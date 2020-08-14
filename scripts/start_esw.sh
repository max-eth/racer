BASE_COMMAND='python -m racer.methods.evolution_strategy_walk.evolution_strategy_walk -F'
WITH_PREFIX='with car_racing_env.headless=True parallel=False iterations=400'

run() {
    number=$1
    shift
    name=$1
    shift
    for i in `seq $number`; do
      echo bsub -J $name-$i -n 1 -W 5:30 -R "rusage[mem=3000]" $BASE_COMMAND logs/$name-$i $WITH_PREFIX $@ 
    done
}

SIGMAS="0.001 0.05 0.01 0.05 0.1 0.5 1 5 10 50 100"
for sigma in $SIGMAS
do
	run 4 new_sigma_$sigma sigma=$sigma
done

run 4 proportional-0.1 weighting="proportional"
run 4 ranked-0.1 weighting="ranked"

for top_k in 1 3 5 10 20 50 100 200
do
	run 4 $top_k-0.1 weighting="top_k" top_k=$top_k
done

