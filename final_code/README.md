
# SLWise



## 0. Install Libs:
```
conda create --name SLWise python==3.7
conda activate SLWise
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch (pytorch 1.6.0, torchvision 0.7.0)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```
To clone code from this project, say
```
git clone XXX
```

## 1. Model Training in the same cell line
#### Training model for A375 cell:（--save_paths为存储的结果文件）
```
cd train_validation
python train_model.py --a375test 1  --lr 0.0002 --max_epochs 200    --save_paths '/home/intern/SyntheticLethal/SL_project/final_code/Result/Res/A375/A375.csv' --cross cv1 
```
#### Training model for HT29 cell:
```
cd train_validation
python train_model.py --HT29test 1  --lr 0.0002  --max_epochs 200 --save_paths '/home/intern/SyntheticLethal/SL_project/final_code/Result/Res/HT29/HT29.csv' --cross cv1
```
#### Training model for A549 cell:
```
cd train_validation
python train_model.py --a549test 1  --lr 0.0002  --max_epochs 200 --save_paths '/home/intern/SyntheticLethal/SL_project/final_code/Result/Res/A549/A549.csv' --cross cv1
```

## 2. Model Training cross the cell line
#### Training model for A375 cell validate on HT29:

```
cd train_validation
python train_model.py --a375test 1  --test_cell 'HT29'  --save_paths '/home/intern/SyntheticLethal/SL_project/final_code/Result/Res/A375/HT29.csv' --out_test 1 --finetine 1 
```
#### Training model for HT29 cell validate on A549:
```
cd train_validation
python train_model.py --HT29test 1  --save_paths '/home/intern/SyntheticLethal/SL_project/final_code/Result/Res/HT29/A549.csv' --out_test 1 --test_cell 'A549' --finetine 1 
```

#### Training model for A549 cell validate on HT29:
```
cd train_validation
python train_model.py --a549test 1   --save_paths '/home/intern/SyntheticLethal/SL_project/final_code/Result/Res/A549/HT29.csv' --out_test 1 --test_cell 'HT29' --finetine 1 
```
## 3. Get  synthetic lethal pairs from  cell lines(Topk为排名前k个基因对)

#### Train on A549 and test on A549 cell
```


python train_model.py --a549test 1 --inference 1 --test_cell 'A549' --save_paths 'A549.csv' 

```

#### Train on A375 and test on A375 cell
```

python train_model.py --a375test 1 --inference 1  --test_cell 'A375' --save_paths 'A375.csv'

```
#### Train on HT29 and test on HT29 cell
```

python train_model.py --HT29test 1 --inference 1 --test_cell 'HT29' --save_paths 'HT29.csv'
```
#### Train on A549 and test on A375 cell
```


python train_model.py --a549test 1 --inference 1 --test_cell 'A375' --save_paths 'A549_A375.csv'

```

#### Train on A375 and test on A549 cell
```

python train_model.py --a375test 1 --inference 1 --test_cell 'A549' --save_paths 'A375_A549.csv'

```
#### Train on A375 and test on HT29 cell
```

python train_model.py --a375test 1 --inference 1 --test_cell 'HT29' --save_paths 'A375_HT29.csv'
```



