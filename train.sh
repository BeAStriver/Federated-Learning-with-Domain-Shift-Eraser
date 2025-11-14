#!/bin/bash
#python -m venv /data/CYY/envs/torchyy
#source /data/CYY/envs/torchyy/bin/activate
#which python
#pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python --version
echo "=== FDSE Federated Training (Determined AI) ==="
echo "Python: $(which python)"
# 设置根路径为当前目录（或上一级）
export PYTHONPATH=$PYTHONPATH:/run/determined/workdir
python3 experiments/train_fdse.py \
     --data_root data/lry/office_caltech10 \
     --rounds 500 \
     --batch_size 50 \
     --lambda_con 0.1 \
     --tau 0.001 \
     --lr 0.01 \
     --seed 42 \
     --repeats 5 \
     --pretrained \
     --verbose
exit 0
#pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
#python /data/YZY/CTLIB-main/setup.py install

#DD_test0
#python ./main.py --mode distill_basic --dataset MNIST --arch LeNet
#DD_test1
#python ./main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet --distill_lr 0.001
#DD_test2
#python ./main.py --mode distill_basic --dataset MNIST --arch LeNet --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
##DD_test3
#python ./main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet --distill_lr 0.001 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train


# med_organamnist_test1 (Random unknown initialization)1*10*11
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet
# med_organamnist_test1_1 (Random unknown initialization) 1*20*11
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 20
# med_organamnist_test1_2 (Random unknown initialization) 1*50*11
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 50
# med_organamnist_test1_3 (Random unknown initialization) 1*30*11 1%
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30
# med_organamnist_test1_4 (Random unknown initialization) 1*60*11 2%
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 60
# med_organamnist_test1_clip (Random unknown initialization)1*10*11
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet
# med_organamnist_test1_1_clip (Random unknown initialization) 1*20*11
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 20
# med_organamnist_test1_2_clip (Random unknown initialization) 1*50*11
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 50
# med_organamnist_test1_3_clip(Random unknown initialization) 1*30*11 1%
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30
# med_organamnist_test1_4_clip (Random unknown initialization) 1*60*11 2%
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 60
# med_organamnist_test1_5 (Random unknown initialization)
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 1
# med_organamnist_test1_6 (Random unknown initialization) ipc=10
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30
# med_organamnist_test1_7 (Random unknown initialization) ipc=5
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30



# med_organamnist_test2 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organamnist_test2_1 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 10 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organamnist_test2_2 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 20 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organamnist_test2_3 (Fixed known initialization)  1%
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organamnist_test2_4 (Fixed known initialization)  2%
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 60 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organamnist_test2_5 (Fixed known initialization)  ipc=10
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organamnist_test2_6 (Fixed known initialization)  ipc=5
#python main.py --mode distill_basic --dataset OrganAMNIST --arch LeNet --distill_steps 30 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train



# med_organsmnist_test1 (Random unknown initialization)1*10*11
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet
# med_organsmnist_test1_1 (Random unknown initialization) 1*20*11
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 20
# med_organsmnist_test1_2 (Random unknown initialization) 1*30*11
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 30
# med_organsmnist_test1_3 (Random unknown initialization) 1*13*11 1%
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 13
# med_organsmnist_test1_4 (Random unknown initialization) 1*25*11 2%
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 25
# med_organsmnist_test1_5 (Random unknown initialization)
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 1
# med_organsmnist_test1_6 (Random unknown initialization) IPC=10
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 30
# med_organsmnist_test1_7 (Random unknown initialization) IPC=5
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 30

# med_organsmnist_test2 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_1 (Fixed known initialization)  1%
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 13 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_2 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 20 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_3 (Fixed known initialization)  2%
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 25 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_4 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 30 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_5 (Fixed known initialization)
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 5 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_6 (Fixed known initialization)ipc=10
#python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 30 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
# med_organsmnist_test2_7 (Fixed known initialization)ipc=5
python main.py --mode distill_basic --dataset OrganSMNIST --arch LeNet --distill_steps 30 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train
