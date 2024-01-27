`uname -srm` yields "Linux 4.18.0-425.10.1.el8_7.x86_64 x86_64".

`lspci | grep -i nvidia` yields the following.
01:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
23:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
23:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
41:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
41:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
61:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
61:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
81:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
81:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
a1:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
a1:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
c1:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
c1:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)
e1:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
e1:00.1 Audio device: NVIDIA Corporation GA102 High Definition Audio Controller (rev a1)

`nvidia-smi` yields the following.
Sat Jan 27 00:52:39 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A6000               Off | 00000000:61:00.0 Off |                  Off |
| 30%   23C    P8              32W / 300W |     16MiB / 49140MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     15655      G   /usr/libexec/Xorg                             8MiB |
+---------------------------------------------------------------------------------------+

`python --version` yields "Python 3.11.4".

Run the following commands.
`pip install arviz`
`pip install ISLP`
`pip install pymc`
`pip install pymc-bart`
`pip install jax`
`pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
`pip install numpyro`

Run `python Predict_Response_Values.py Bayesian_Neural_Network 5000 Feature_Matrix_Of_Docking_Scores_And_Numbers_Of_Occurrences_Of_Substructures.csv Docking_Score`.
