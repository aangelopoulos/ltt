conda env create -f environment.yml
conda activate fdrcps
cd ./experiments/coco/src/ASL/
mkdir models_local
cd models_local
wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TResNet_xl_640_88.4.pth 
cd ../
