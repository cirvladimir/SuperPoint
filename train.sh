export TMPDIR=/tmp
export PYTHONPATH=$(pwd)
cd superpoint
while true
do
  python3 experiment.py train configs/magic-point_shapes.yaml magic-point_synth
done
