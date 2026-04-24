conda create -n xicm_5090 python=3.11 -y
conda activate xicm_5090

# 先装 torch/vllm 5090 新栈
# 然后：
./setup_xicm_env.sh

source env_coppelia.sh
bash scripts/eval_XICM.sh "0" 1 Qwen2.5.7B.instruct 1 0 "random" false