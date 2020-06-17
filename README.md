# Fork from https://github.com/liyaguang/DCRNN and update with my experiments
# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.

The Pytorch implementaion of the model is available at [DCRNN-Pytorch](https://github.com/chnsh/DCRNN_PyTorch).

## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- pyaml
- statsmodels
- tensorflow>=1.3.0


Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/` folder.
The `*.h5` files store the data in `panads.DataFrame` using the `HDF5` file format. Here is an example:

|                     | sensor_0 | sensor_1 | sensor_2 | sensor_n |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|
| 2018/01/01 00:00:00 |   60.0   |   65.0   |   70.0   |    ...   |
| 2018/01/01 00:05:00 |   61.0   |   64.0   |   65.0   |    ...   |
| 2018/01/01 00:10:00 |   63.0   |   65.0   |   60.0   |    ...   |
|         ...         |    ...   |    ...   |    ...   |    ...   |


Here is an article about [Using HDF5 with Python](https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773).

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Graph Construction
 As the currently implementation is based on pre-calculated road network distances between sensors, it currently only
 supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`).
```bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```
Besides, the locations of sensors in Los Angeles, i.e., METR-LA, are available at [data/sensor_graph/graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv), and the locations of sensors in PEMS-BAY are available at [data/sensor_graph/graph_sensor_locations_bay.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations_bay.csv).

## Model Training
```bash
# METR-LA
python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-BAY
python dcrnn_train.py --config_filename=data/model/dcrnn_bay.yaml
```
Each epoch takes about 5min or 10 min on a single GTX 1080 Ti for METR-LA or PEMS-BAY respectively. 

There is a chance that the training loss will explode, the temporary workaround is to restart from the last saved model before the explosion, or to decrease the learning rate earlier in the learning rate schedule. 


## Eval baseline methods
```bash
# METR-LA
python -m scripts.eval_baseline_methods --traffic_reading_filename=data/metr-la.h5
```
More details are being added ...

## !! Update with my experiments' results (on METR-LA dataset)
All of experiments' results are in folder `data/model/`

Folder structure: `data_[DR: dual_random_walk filter, R: random_walk filter, L: laplacian filter, I: identity filter]_h_[horizon, default = 12]_...`

In each folder, see the model hyperparameters and training configuration in file `.yaml` which also has the trained model checkpoint for testing.

The experiments' results with testing data are in `info.log` file in each training folder. You can see the comparison results in my Presentation file!

## Run the Pre-trained Model on METR-LA

```bash
# METR-LA
python run_demo.py --config_filename=data/model/pretrained/METR-LA/config.yaml

# PEMS-BAY
python run_demo.py --config_filename=data/model/pretrained/PEMS-BAY/config.yaml
```
The generated prediction of DCRNN is in `data/results/dcrnn_predictions`.

## !! Update with a crowd flow dataset - Beijing taxi trajectories in 2014
Folder `data/beijing2014` with 2 files: 
- `taxi_flow.csv`: The data of outflow and inflow of each taxi for each region at each timestamp

Example data:
```
      time region outflow inflow
  1    1      1       3      2
  2    2      1       0      6
  3    3      1       1      3
```
- `taxi_timemap.csv`: The mapping from each timestamp to real time, as well as dayinweek, hourinday, hourinweek, and the week since the begining of the dataset.
Note that dayinweek is denoted as follows:
```Mon -> Friday: 1 -> 5
Saturday: 6
Sunday: 7
```

Example data:
```
    time            datetime dayinweek hourinday hourinweek week
  1    1 2014-04-01 00:00:00         2         1         25    0
  2    2 2014-04-01 01:00:00         2         2         26    0
  3    3 2014-04-01 02:00:00         2         3         27    0
  4    4 2014-04-01 03:00:00         2         4         28    0
```

## Deploying DCRNN on Large Graphs with graph partitioning

With graph partitioning, DCRNN has been successfully deployed to forecast the traffic of the entire California highway network with **11,160** traffic sensor locations simultaneously. The general idea is to partition the large highway network into a number of small networks, and trained them with a share-weight DCRNN simultaneously. The training process takes around 3 hours in a moderately sized GPU cluster, and the real-time inference can be run on traditional hardware such as CPUs.

See the [paper](https://arxiv.org/pdf/1909.11197.pdf "GRAPH-PARTITIONING-BASED DIFFUSION CONVOLUTION RECURRENT NEURAL NETWORK FOR LARGE-SCALE TRAFFIC FORECASTING"), [slides](https://press3.mcs.anl.gov/atpesc/files/2019/08/ATPESC_2019_Track-8_11_8-9_435pm_Mallick-DCRNN_for_Traffic_Forecasting.pdf), and [video](https://www.youtube.com/watch?v=liJNNtJGTZU&list=PLGj2a3KTwhRapjzPcxSbo7FxcLOHkLcNt&index=10) by Tanwi Mallick et al. from Argonne National Laboratory for more information.


## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
