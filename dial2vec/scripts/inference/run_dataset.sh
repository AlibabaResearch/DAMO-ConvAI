models=('bert' 'plato' 'roberta' 'todbert' 'blender' 'simcse' 't5')

dataset=$1

for model in ${models[*]}
do
  sh scripts/inference/run_${model}.sh ${dataset}
done