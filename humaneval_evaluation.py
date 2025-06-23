#!/usr/bin/env python3
"""
HumanEval evaluation script with Docker sandbox
"""
import json
import subprocess
import tempfile
import os
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import docker
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanEvalEvaluator:
    def __init__(self, timeout: int = 10, max_workers: int = 4):
        self.timeout = timeout
        self.max_workers = max_workers
        self.docker_client = None
        self.setup_docker()
    
    def setup_docker(self):
        """Setup Docker client and evaluation image"""
        try:
            self.docker_client = docker.from_env()
            # self.build_evaluation_image()
            self.docker_client.images.get("humaneval-sandbox")
        except Exception as e:
            logger.error(f"Docker setup failed: {e}")
            raise
    
    
    def load_results(self, results_file: str) -> List[Dict]:
        """Load inference results"""
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))

        ### use canonical_solution to test whether the rest of implementation is correct
        # results = []
        # from datasets import load_dataset
        # dataset = load_dataset("openai/openai_humaneval")
        # for item in dataset["test"]:
        #     results.append({
        #         "task_id": item["task_id"],
        #         "completion": item["canonical_solution"] 
        #     })

        return results
    
    def load_humaneval_tests(self) -> Dict[str, str]:
        """Load HumanEval test cases"""
        from datasets import load_dataset
        dataset = load_dataset("openai/openai_humaneval")
        
        tests = {}
        for item in dataset["test"]:
        
            tests[item["task_id"]] = {
                "prompt": item["prompt"],
                "test":item["test"],
                "entry_point": item["entry_point"]
            }
        
        return tests
    
    def execute_code_in_docker(self, code: str, test_item: Dict) -> Tuple[bool, str]:
        """Execute code safely in Docker container"""
        import textwrap
        try:
            full_code= f"""
{test_item["prompt"]}
{code}
{test_item["test"]}
check({test_item["entry_point"]})
print("PASSED")
"""
            # Start container
            container = self.docker_client.containers.run(
                "humaneval-sandbox",
                "sleep infinity",
                detach=True,
                mem_limit="128m",
                cpu_period=100000,
                cpu_quota=50000
            )

            try:
                # Execute code and get immediate results
                exec_result = container.exec_run(
                    ["python", "-c", full_code]
                )

                logs = exec_result.output.decode('utf-8')
                success = "PASSED" in logs and exec_result.exit_code == 0
                
                print(f"Exit code: {exec_result.exit_code}")
                print(f"Output: {logs}")
                
                return success, logs
                
            finally:
                container.remove(force=True)
                
        except Exception as e:
            logger.error(f"Error executing code in Docker: {e}")
            return False, str(e)
    
    def evaluate_single_solution(self, solution: Dict, tests: Dict[str, str]) -> Dict:
        """Evaluate a single solution"""
        task_id = solution["task_id"]
        completion = solution["completion"]
        
        if task_id not in tests:
            return {
                "task_id": task_id,
                "sample_id": solution.get("sample_id", 0),
                "passed": False,
                "error": "Test not found"
            }
        
        test_item = tests[task_id]
        passed, logs = self.execute_code_in_docker(completion, test_item)
        
        return {
            "task_id": task_id,
            "sample_id": solution.get("sample_id", 0),
            "passed": passed,
            "logs": logs[:500] if logs else ""  # Truncate logs
        }
    
    def evaluate_all(self, results_file: str, output_file: str) -> Dict:
        """Evaluate all solutions"""
        logger.info("Loading results and tests...")
        solutions = self.load_results(results_file)
        tests = self.load_humaneval_tests()
        
        logger.info(f"Evaluating {len(solutions)} solutions...")
        
        evaluation_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_solution = {
                executor.submit(self.evaluate_single_solution, solution, tests): solution
                for solution in solutions
            }
            
            for future in tqdm(as_completed(future_to_solution), total=len(solutions), desc="Evaluating"):
                try:
                    result = future.result()
                    evaluation_results.append(result)
                except Exception as e:
                    solution = future_to_solution[future]
                    logger.error(f"Error evaluating {solution['task_id']}: {e}")
                    evaluation_results.append({
                        "task_id": solution["task_id"],
                        "sample_id": solution.get("sample_id", 0),
                        "passed": False,
                        "error": str(e)
                    })
        
        # Calculate metrics
        metrics = self.calculate_metrics(evaluation_results)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump({
                "results": evaluation_results,
                "metrics": metrics
            }, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {output_file}")
        logger.info(f"Pass@1: {metrics['pass_at_1']:.2%}")
        
        return metrics
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        # Group by task_id
        task_results = {}
        for result in results:
            task_id = result["task_id"]
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(result["passed"])
        
        # Calculate pass@k metrics
        total_tasks = len(task_results)
        pass_at_1 = sum(1 for results in task_results.values() if any(results)) / total_tasks
        
        # If multiple samples per task, calculate pass@k for k > 1
        max_samples = max(len(results) for results in task_results.values())
        
        metrics = {
            "pass_at_1": pass_at_1,
            "total_tasks": total_tasks,
            "total_solutions": len(results),
            "max_samples_per_task": max_samples
        }
        
        # Calculate pass@k for k > 1 if applicable
        if max_samples > 1:
            for k in [2, 3, 5, 10, 20]:
                if k <= max_samples:
                    pass_at_k = self.calculate_pass_at_k(task_results, k)
                    metrics[f"pass_at_{k}"] = pass_at_k
        
        return metrics
    
    def calculate_pass_at_k(self, task_results: Dict, k: int) -> float:
        """Calculate pass@k metric"""
        total = 0
        passed = 0
        
        for results in task_results.values():
            if len(results) >= k:
                total += 1
                if any(results[:k]):
                    passed += 1
        
        return passed / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate HumanEval results")
    parser.add_argument("--results", required=True, help="Results file from inference")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file")
    parser.add_argument("--timeout", type=int, default=10, help="Execution timeout seconds")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    evaluator = HumanEvalEvaluator(args.timeout, args.max_workers)
    metrics = evaluator.evaluate_all(args.results, args.output)
    
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        if key.startswith("pass_at_"):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()