srun --job-name jupyter -p gpu --time 06:00:00 bash -c 'jupyter notebook --ip $(hostname -i) --no-browser'

# Forward in console on laptop using ssh -N -L localhost:8888:<ip>:8888 <tenant-alias>