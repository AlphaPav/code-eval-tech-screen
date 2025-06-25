#!/usr/bin/env python3
"""
HumanEval inference script for Qwen2.5-Coder model
"""
import json
import requests
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from typing import List, Dict, Optional
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    # "\nprint(",
    # "\n#",
    "\n```",
]



class HumanEvalInference:
    def __init__(self, api_base: str = "http://localhost:8080", model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.api_base = api_base
        self.model_name = model_name
        self.session = requests.Session()
        
    def load_humaneval_dataset(self) -> List[Dict]:
        """Load HumanEval dataset"""
        logger.info("Loading HumanEval dataset...")
        dataset = load_dataset("openai/openai_humaneval")
        return list(dataset["test"])
    
    def create_prompt(self, problem: Dict, strategy: str = "direct") -> str:
        """Create prompt for code generation"""
        prompt = problem["prompt"]
        
        if strategy == "direct":
            return prompt
        if strategy == "instruct":
            return f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""

        elif strategy == "cot":
            return f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function? Think step by step and then provide the complete solution.

```python
{prompt}
```
<|im_end|><|im_start|>assistant
# Let me think step by step:
# 1. Understand what the function needs to do
# 2. Plan the implementation
# 3. Write the code
```python
"""
        elif strategy == "examples":
            return f'''<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|><|im_start|>user
Here are some example solutions:

```python
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b
```

```python
def reverse_string(s):
    """Reverse a string."""
    return s[::-1]
```

```python
def find_max(numbers):
    """Find maximum in a list."""
    if not numbers:
        return None
    return max(numbers)
```

Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
'''            

        else:
            return prompt
    
    def generate_code(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512, 
                     num_samples: int = 1) -> List[str]:
        """Generate code completion using vLLM API"""
        try:
            if num_samples == 1:
                response = self.session.post(
                    f"{self.api_base}/v1/completions",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": EOS,
                        "echo": False  # This prevents echoing the prompt back
                    },
                    timeout=60
                )
            else:
                response = self.session.post(
                    f"{self.api_base}/v1/completions",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "n": num_samples,
                        "stop": EOS,
                        "echo": False  # This prevents echoing the prompt back
                    },
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                if num_samples == 1:
                    return [result["choices"][0]["text"]]
                else:
                    return [choice["text"] for choice in result["choices"]]
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return [""]
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return [""]
    
    def process_single_problem(self, problem: Dict, config: Dict) -> Dict:
        """Process a single HumanEval problem"""
        task_id = problem["task_id"]
        prompt = self.create_prompt(problem, config.get("strategy", "direct"))
        
        completions = self.generate_code(
            prompt, 
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 512),
            num_samples=config.get("num_samples", 1)
        )
        
        results = []
        for i, completion in enumerate(completions):
            # full_code = problem["prompt"] + completion
            results.append({
                "task_id": task_id,
                "completion": completion,
                "sample_id": i
            })
        
        return results
    
    def run_inference(self, output_file: str, config: Dict = None) -> None:
        """Run inference on entire HumanEval dataset"""
        if config is None:
            config = {
                "temperature": 0,
                "max_tokens": 2048,
                "num_samples": 1,
                "strategy": "instruct",
                "max_workers": 4
            }
        
        logger.info("Starting HumanEval inference...")
        logger.info(f"Configuration: {config}")
        
        # Load dataset
        problems = self.load_humaneval_dataset()
        logger.info(f"Loaded {len(problems)} problems")
        
        all_results = []
        
        # Process problems with threading for efficiency
        with ThreadPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
            future_to_problem = {
                executor.submit(self.process_single_problem, problem, config): problem 
                for problem in problems
            }
            
            for future in tqdm(as_completed(future_to_problem), total=len(problems), desc="Processing"):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    problem = future_to_problem[future]
                    logger.error(f"Error processing {problem['task_id']}: {e}")
        
        # Save results
        logger.info(f"Saving {len(all_results)} results to {output_file}")
        with open(output_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
        
        logger.info("Inference completed!")

def main():
    parser = argparse.ArgumentParser(description="Run HumanEval inference")
    parser.add_argument("--api-base", default="http://localhost:8080", help="vLLM API base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Model name")
    parser.add_argument("--output", default="humaneval_results.jsonl", help="Output file")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--strategy", choices=["direct", "instruct", "cot", "examples"], default="instruct", help="Prompting strategy")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_samples": args.num_samples,
        "strategy": args.strategy,
        "max_workers": args.max_workers
    }
    
    inference = HumanEvalInference(args.api_base, args.model)
    inference.run_inference(args.output, config)

if __name__ == "__main__":
    main()