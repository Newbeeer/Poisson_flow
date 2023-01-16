#srun --job-name jupyter --gpus=1 --gres=gpumem:16G --time 06:00:00 bash -c 'jupyter notebook --ip $(hostname -i) --no-browser'
srun --job-name jupyter --time 06:00:00 bash -c 'jupyter notebook --ip $(hostname -i) --no-browser'

# Forward in console on laptop using ssh -N -L localhost:8888:<ip>:8888 <tenant-alias>