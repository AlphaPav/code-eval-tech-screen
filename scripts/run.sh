
## change the temperature 
strategy=direct
temp=0.2


python humaneval_inference.py \
    --output results/${strategy}_results_temp${temp}.jsonl \
    --temperature ${temp} \
    --num-samples 20 \
    --max-tokens 512 \
    --strategy ${strategy}


python humaneval_evaluation.py \
    --results results/${strategy}_results_temp${temp}.jsonl  \
    --output results/evaluation_${strategy}temp${temp}.json \
    --max-workers 12


cat results/evaluation_${strategy}temp${temp}.json | jq '.metrics'


 

