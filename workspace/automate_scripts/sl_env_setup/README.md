
### Data Split
#TODO: generalize the data split generation configs
#TODO: add instructions in README under root folder

Since HIGGS dataset is already randomly recorded,
data split will be specified by the continuous index ranges for each client,
rather than a vector of random instance indices.
We provide four options to split the dataset to simulate the non-uniformity in data quantity: 

1. uniform: all clients has the same amount of data 
2. linear: the amount of data is linearly correlated with the client ID (1 to M)
3. square: the amount of data is correlated with the client ID in a squared fashion (1^2 to M^2)
4. exponential: the amount of data is correlated with the client ID in an exponential fashion (exp(1) to exp(M))

The choice of data split depends on dataset and the number of participants.

For a large dataset like HIGGS, if the number of clients is small (e.g. 5),
each client will still have sufficient data to train on with uniform split,
and hence exponential would be used to observe the performance drop caused by non-uniform data split.
If the number of clients is large (e.g. 20), exponential split will be too aggressive, and linear/square should be used.

Data splits used in this example can be generated with
```
bash data_split_gen.sh
```

This will generate data splits for three client sizes: 2, 5 and 20, and 3 split conditions: uniform, square, and exponential.
If you want to customize for your experiments, please check `utils/prepare_data_split.py`.

> **_NOTE:_** The generated train config files will be stored in the folder `/tmp/nvflare/xgboost_higgs_dataset/`,
> and will be used by job_configs by specifying the path within `config_fed_client.json` 