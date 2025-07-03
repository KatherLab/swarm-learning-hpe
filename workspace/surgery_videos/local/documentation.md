# Documentation Surgery Swarm
## Prepare workspace
```
cd Desktop/surgery_swarm
source /home/swarm/Desktop/surgery_swarm/.venv/bin/activate
```
## Training
> Change output folder for every experiment run! 
```
python3 -m training [-b] -o /mnt/sda1/surgery_swarm/output_01 -d /mnt/sda1/surgery_swarm/data
```
- `-b`: Binary flag if set only calss 0/1
- `-o`: Output folder e.g. `/mnt/sda1/surgery_swarm/validation_01_center04`
- `-d`: Data folder e.g `/mnt/sda1/surgery_swarm/data`
## Validation
> Change output folder for every experiment run! 
```
python3 -m validation -e [-b] -o /mnt/sda1/surgery_swarm/validation_01_center04 -d /mnt/sda1/surgery_swarm/data -m /mnt/sda1/surgery_swarm/output_01
```
- `-b`: Binary flag if set only calss 0/1
- `-o`: Output folder e.g. `/mnt/sda1/surgery_swarm/validation_01_center04`
- `-d`: Data folder e.g `/mnt/sda1/surgery_swarm/data`
- `-m`: Model folder e.g. `/mnt/sda1/surgery_swarm/output_01`

## Clean GPU
```
nvidia-smi
kill [PID]
```