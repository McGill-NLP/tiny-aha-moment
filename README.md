# Recreating R1's Aha Moment
Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero), but designed to be 10X simpler, cleaner, and faster.

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone git@github.com:McGill-NLP/tiny-aha-moment.git
   ```

2. **Install dependencies** 

    **option 1: with pip**

    First, load the necessay cuda tools:
    ```bash
    module load cudatoolkit/12.5
    ```  
    Next, install torch:  
    ```bash
    pip install torch==2.5
    ```  
    Next, follow the installation guide on the [vllm website](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html) for installing vllm. 
    ```bash
    pip install vllm
    ```  
    Next,
    ```bash
    pip install datasets deepspeed jupyter ipykernel ipywidgets wandb
    ``` 
    Next, install flash attention,
    ```bash
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ``` 
    You should be set.

    **option 2: with uv**

    If necessary, install `uv`, [see their website](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

    First, load the necessay cuda tools:
    ```bash
    module load cudatoolkit/12.4
    ```

    Install the environment in `.venv`
    ```bash
    uv sync
    ```

    Install flash-attention
    ```bash
    uv sync --extra compile
    ```  

    You can run with either `uv run r1_script.py` or activate the env with `source .venv/bin/activate` and run with `python r1_script.py`
    
    

3. **Start an interactive job on the cluster**  
   Request resources using:  
   ```bash
   salloc --partition=main --gres=gpu:a100l:1 -c 6 --mem=64G -t 12:00:00
   ```  
   Then, connect via VS Code or Cursor.

4. **Run the training script**  
   Open `r1_gold.ipynb`, set `CUDA_HOME` and `HF_HOME` as needed, and start training.

5. **Install VS Code extensions**  
   Make sure to install the **Jupyter** and **Python** extensions in VS Code for a smoother experience.

## File Descriptions
- `r1_gold.ipynb` is the ground truth implementation.
- `r1_todo.ipynb` misses some components and you need to fill those without looking at the `r1_gold.ipynb`.
- `r1_script.py` is also just the `r1_gold` but for convenience of running with python.
