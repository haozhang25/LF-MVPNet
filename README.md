# LF-MVPNet: Consistent Light Field Low-Light Enhancement via MPI-driven View Propagation

## Dependencies
* Pytorch 2.0.1
* CUDA 11.7
* Python 3.10
* Matlab(For data generation)
## Prepare Training and Test Data
* To generate the training data, please first download the L3F dataset and run:
  ```
  GenerateMatData.m
  GenerateDataForTraining.m
  ```
* To generate the test data, run:
  ```
  GenerateMatData.m
  GenerateDataForTest.m
  ```
## Train
* Run:
  ```
  python train.py
  ```
## Test
* Run:
  ```
  python test.py
  ```
  
## Acknowledgement
Our work and code implementation build upon the following prior work:

- [PFInet](https://github.com/XinLuo0/PFInet)