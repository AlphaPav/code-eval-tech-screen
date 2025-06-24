
## change the temperature 
strategy=instruct
for temp in 0.2 0; do


python humaneval_inference.py \
    --output results/${strategy}_results_temp${temp}.jsonl \
    --temperature ${temp} \
    --num-samples 1 \
    --max-tokens 2048 \
    --strategy ${strategy}


python humaneval_evaluation.py \
    --results results/${strategy}_results_temp${temp}.jsonl  \
    --output results/evaluation_${strategy}temp${temp}.json \
    --max-workers 12

cat results/evaluation_${strategy}temp${temp}.json | jq '.metrics'
done 



 

