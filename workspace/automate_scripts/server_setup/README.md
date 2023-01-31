![Screenshot from 2023-01-31 14-02-44.png](..%2F..%2F..%2Fassets%2FScreenshot%20from%202023-01-31%2014-02-44.png)

It could be very tricky to set up the gpu env for SL. 
First make sure 
- [] `$ nvidia-smi` returns the driver status. `$ nvcc -V` doesn't need to be set up. 
- If all the debugging fails, try to `$ sudo apt-get purge -y <all nvidia drivers/environments>`
- Reinstall nvidia driver as the picture shown
- Rerun the gpu_env_setup.sh
