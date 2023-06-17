wget https://us.download.nvidia.com/tesla/525.105.17/NVIDIA-Linux-x86_64-525.105.17.run
sudo sh NVIDIA-Linux-x86_64-525.105.17.run

# To check that it worked: sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

# Run this on your laptop:
# scp 35.184.243.45:.ssh/id_ecdsa.pub a
# scp a 35.223.51.78:.ssh/id_ecdsa.pub
# rm a
# cat .ssh/id_ecdsa.pub >> .ssh/authorized_keys
# sudo chmod -R 0700 .ssh
