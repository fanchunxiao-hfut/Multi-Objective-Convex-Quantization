# Multi-Objective Convex Quantization for Efficient Model Compression
This is an implementation of EQ, an efficient quantization framework with quantization and self-distillation. 

## Requirement
```
python>=3.6
pytorch>=1.9.0
```
## How to use
Take the example of running a 4bit resnet20-cifar10
```
cd ./EQ_code
python main.py --data_type cifar10 --data_path ./dataset --model resnet20_cifar10 --q_bit 4 --a_bit 4 --save_dir ./save_model_r20_c10_4bit
```
The main.py file is used to train the full-precision model, and the weight distribution of the trained full-precision model is similar to the weight distribution of the quantized model of the corresponding bit. 


The trained model also needs to be converted into a quantized model and the accuracy of the quantized model tested:
```
python quant_test.py --pretrainedModel./save_model_r20_c10_4bit/model_name.pth --q_bit 4
```

We provide some of our trained quantized models
```
./quantized_model/resnet20_cifar10_q2_acc_0.9148.pth
./quantized_model/resnet20_cifar10_q3_acc_0.9319.pth
```
