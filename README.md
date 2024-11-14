# toyRL
Toy examples to get a feeling of RL

# environment setup 
conda create -n toyrl python=3.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy==1.23.5

# install gym dependencies
pip install gymnasium tqdm
pip install "gymnasium[classic-control]"

pip install matplotlib

# install isaacgym
cd /home/zby/Programs/isaacgym/python
pip install -e .

# install isaacgymenvs
cd /home/zby/Programs/IsaacGymEnvs
pip install -e .

cd toyRL
pip install -e .