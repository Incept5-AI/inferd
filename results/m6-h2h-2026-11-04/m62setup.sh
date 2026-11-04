#!/bin/zsh
# m6-2 setup: front-end venv (Python 3.13 like mini-m6's mlxvlm venv) and the DFlash2 drafter in the HF cache layout
export PATH=$HOME/.local/bin:$PATH
uv venv ~/.venvs/mlxvlm --python 3.13 > ~/inference/setup.log 2>&1
VIRTUAL_ENV=~/.venvs/mlxvlm uv pip install -r ~/inference/rel020/engine/requirements.txt >> ~/inference/setup.log 2>&1
VIRTUAL_ENV=~/.venvs/mlxvlm uv pip install "huggingface_hub[cli]" >> ~/inference/setup.log 2>&1
~/.venvs/mlxvlm/bin/hf download incoai/Qwen3.8-27B-DFlash2 --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 >> ~/inference/setup.log 2>&1
echo "setup exit $?" >> ~/inference/setup.log; touch ~/inference/SETUP_DONE
