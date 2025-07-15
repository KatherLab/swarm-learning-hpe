## Odelia Swarm Learning Nodes Overview

Nodes will be added to the VPN and will be able to communicate with each other after setting up the Swarm Learning Environment with [Install](#install)

| Project       | Node Name | Location               | Hostname  | Data | Maintainer                                |
| ------------- | --------- | ---------------------- | --------- | -----| ----------------------------------------- |
| Sentinel node | TUD       | Dresden, Germany       | swarm     |      | [@Jeff](https://github.com/Ultimate-Storm) |
| ODELIA        | VHIO      | Madrid, Spain          | radiomics |      | [Adrià](mailto:adriamarcos@vhio.net)       |
|               | UKA       | Aachen, Germany        | swarm     |      | [Gustav](mailto:gumueller@ukaachen.de)     |
|               | RADBOUD   | Nijmegen, Netherlands  | swarm     |      | [Tianyu](mailto:t.zhang@nki.nl)            |
|               | MITERA    | Paul, Greece           |           |      |                                             |
|               | RIBERA    | Lopez, Spain           |           |      |                                             |
|               | UTRECHT   |                        |           |      |                                             |
|               | CAMBRIDGE | Nick, Britain          |           |      |                                             |
|               | ZURICH    | Sreenath, Switzerland  |           |      |                                             |
| SWAG          |           |                        | swarm     |      |                                             |
| DECADE        |           |                        | swarm     |      |                                             |
## Models implemented

TUD benchmarking on Duke breast mri dataset:![TUD experiments result.png](assets%2FTUD%20experiments%20result.png)

Report: [Swarm learning report.pdf](assets%2FSwarm%20learning%20report.pdf)

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

Following these structured steps will help in maintaining a well-organized dataset, thereby enhancing data management and processing in your projects.
