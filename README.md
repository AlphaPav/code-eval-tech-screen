# code-eval-tech-screen


## Setup Environment
```
conda create -n codeeval python=3.10 -y
source activate codeeval
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo apt-get install jq 
pip install -r requirements.txt
```


## Serving the Model with vLLM
Utilize the vLLM to serve the model on the CPU or GPU.
Create a Python script to set up a Docker instance that serves the model. This will ensure the environment is consistent and portable.


- Start the vLLM: 
```
bash scripts/start_vllm.sh
```

- To test if it's successful:
```
python test_vllm_serving.py
```


## Inference:
Develop a script to perform inference the HumanEval dataset.
This script should interact with the served model to generate predictions for the provided samples.

- Run the inference:
```
python humaneval_inference.py \
    --output results/instruct_results_temp0.2.jsonl \
    --temperature 0.2 \
    --num-samples 1 \
    --max-tokens 2048 \
    --strategy instruct
```

## Evaluation:
Use a sandbox environment (docker instance) to assess the pass rate of the HumanEval results obtained from the Qwen2.5-Coder-0.5B-Instruct model.
This evaluation will help determine the effectiveness of the model's predictions.

- Build HumanEval Docker sandbox image with sudo

```
sudo bash scripts/eval_docker.sh
```

- Execute the code and assess the pass rate

```
python humaneval_evaluation.py \
    --results results/instruct_results_temp0.2.jsonl \
    --output results/evaluation_temp0.2.json
```


- Print the results
```
cat results/evaluation_temp0.2.json | jq '.metrics'
```

### Run inference and evaluation together
use 
```
bash script/run.sh 
```

### Results

direct prompt

| Temperature | Pass@1 (%) |
|-------------|------------|
| 0.0         | 44.51      |
| 0.2         | 45.73      |


chat template

| Temperature | Pass@1 (%) |
|-------------|------------|
| 0.0         | 59.76      |
| 0.2         | 58.54      |


## Performance & Quality Improvement
### How can you improve the HumanEval’s metric? Be open-minded.

- Pass@k with k>1: Generate multiple samples (e.g., k=5, 10, 20) per problem
- Best-of-N: Generate N samples, select best based on confidence/length

### How can you enhance the performance of the inference and evaluation processes.

- Explore multiple sampling strategies.

```
# Generate multiple samples per problem
python humaneval_inference.py \
    --num-samples 10 \
    --temperature 0.8 \
    --output results_multi.jsonl

# Evaluate with pass@k metrics
python humaneval_evaluation.py \
    --results results_multi.jsonl \
    --output evaluation_multi.json
```

- Explore multiple prompting strategies.

```
# Try different prompting strategies
python humaneval_inference.py --strategy examples # With examples
```

- Self-reflection and refinement/test-time-compute. For example, use the following prompt for the generated code and do the inference again:
```
# Previous attempt had issues. Let me improve it:

{problem['prompt']}

# Previous attempt:
{previous_attempt}

# Issues to fix:
# 1. Check edge cases
# 2. Verify logic correctness
# 3. Ensure proper return type

# Improved solution:
```


- To enhance the evaluation process,  we can also do some smart filtering, e.g., 
```
# Quick syntax validation before execution
def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
```

### How can you scale this evaluation process and make it run faster?

We can add more workers for parallel processing.

```
# Use more workers for parallel processing
python humaneval_inference.py --max-workers 8
python humaneval_evaluation.py --max-workers 16
```