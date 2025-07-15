# ODELIA consortium

## Usage
### Ensuring Dataset Structure

To ensure proper organization of your dataset, please follow the steps outlined below:

1. **Directory Location**

   Place your dataset under the specified path:

/workspace/odelia-breast-mri/user/data-and-scratch/data


Within this path, create a folder named `multi_ext`. Your directory structure should then resemble:
/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user/data-and-scratch/data
└── multi_ext
├── datasheet.csv # Your clinical tabular data
├── test # External validation dataset
├── train_val # Your own site training data
└── segmentation_metadata_unilateral.csv # External validation table

2. **Data Organization**

Inside the `train_val` or `test` directories, place folders that directly contain NIfTI files. The folders should be named according to the following convention:

<patientID>_right
<patientID>_left

Here, `<patientID>` should correspond with the patient ID in your tables (`datasheet.csv` and `segmentation_metadata_unilateral.csv`). This convention assists in linking the imaging data with the respective clinical information efficiently.

#### Summary

- **Step 1:** Ensure your dataset is placed within `/workspace/odelia-breast-mri/user/data-and-scratch/data/multi_ext`.
- **Step 2:** Organize your clinical tabular data, external validation dataset, your own site training data, and external validation table as described.
- **Step 3:** Name folders within `train_val` and `test` as `<patientID>_right` or `<patientID>_left`, matching the patient IDs in your datasheets.

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


## Node list

Nodes will be added to vpn and will be able to communicate with each other after setting up the Swarm Learning Environment with [Install](#install)
| Project | Node Name | Location           | Hostname  | Data      | Maintainer                                 |
| ------- | --------- | ------------------| ---------| --------- | ------------------------------------------|
| Sentinel node | TUD       | Dresden, Germany  | swarm     | | [@Jeff](https://github.com/Ultimate-Storm) |
| ODELIA  | VHIO      | Madrid, Spain      | radiomics |  | [@Adrià](adriamarcos@vhio.net)           |
|         | UKA       | Aachen, Germany    | swarm     |   | [@Gustav](gumueller@ukaachen.de)         |
|         | RADBOUD   | Nijmegen, Netherlands | swarm |   | [@Tianyu](t.zhang@nki.nl)                |
|         | MITERA    | Paul, Greece  |           |   |                                            |
|         | RIBERA    | Lopez, Spain    |           |   |                                            |
|         | UTRECHT   |                    |           |  |                                            |
|         | CAMBRIDGE |  Nick, Britain  |           |  |                                            |
|         | ZURICH    |  Sreenath, Switzerland   |           |           |                                            |
| SWAG |        |   | swarm     |      |  |

| DECADE |        |   | swarm     |      | |


| Other nodes | UCHICAGO  | Chicago, USA       | swarm     |           | [@Sid](Siddhi.Ramesh@uchospitals.edu)    |