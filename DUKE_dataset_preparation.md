## Data Preparation
### Notes
This will take a long time. Just run with either command, `get_dataset_gdown.sh` is recommended to run before you have done step 2, `get_dataset_scp.sh` is recommended to run after you have done step 2.
`get_dataset_gdown.sh` will download the dataset from Google Drive.
```sh
$ sh workspace/automate_scripts/sl_env_setup/get_dataset_gdown.sh
```
The [-s sentinel_ip] flag is only necessary for `get_dataset_scp.sh` The script will download the dataset from the sentinel node.
```sh
$ sh workspace/automate_scripts/sl_env_setup/get_dataset_scp.sh -s <sentinel_ip>
```

### Instructions

1. Make sure you have downloaded Duke data.
    
2. Create the folder `WP1` and in it `test` and `train_val`
```bash
mkdir workspace/<workspace-name>/user/data-and-scratch/data/WP1
mkdir workspace/<workspace-name>/user/data-and-scratch/data/WP1/{test,train_val}
```
3. Search for your institution in the [Node list](#nodelist) and note the data series in the column "Data"

4. Prepare the clinical tables
```sh
cp workspace/<workspace-name>/user/data-and-scratch/data/*.xlsx workspace/<workspace-name>/user/data-and-scratch/data/WP1
```

5. Copy the nifty files from feature folder into `WP1/test` from 801 to 922
```sh
cp -r workspace/<workspace-name>/user/data-and-scratch/data/odelia_dataset_only_sub/{801..922}_{right,left} workspace/<workspace-name>/user/data-and-scratch/data/WP1/test
```
   
6. Copy the nifty files from feature folder with the order you noted into `WP1/train_val` from xxx to yyy
```sh
cp -r workspace/<workspace-name>/user/data-and-scratch/data/odelia_dataset_only_sub/{<first_number>..<second_number>} workspace/<workspace-name>/user/data-and-scratch/data/WP1/train_val
```
