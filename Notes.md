# Notes

## vLLM Request Output Structure

Structure of the outputs when making a _request_ with vLLM, for reference. 

```python
RequestOutput(
    request_id=0,
    prompt="The capital of France is",
    prompt_token_ids=[1, 450, 7483, 310, 3444, 338],
    encoder_prompt=None,
    encoder_prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[
        CompletionOutput(
                index=0,
            text=" Paris.",
            token_ids=(3681, 29889, 13),
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=stop,
            stop_reason="\n",
        )
    ],
    finished=True,
    metrics=RequestMetrics(
            arrival_time=1728338377.7350004,
        last_token_time=1728338377.7350004,
        first_scheduled_time=1728338377.73668,
        first_token_time=1728338377.754303,
        time_in_queue=0.0016796588897705078,
        finished_time=1728338377.765628,
        scheduler_time=0.000719655305147171,
        model_forward_time=None,
        model_execute_time=None,
    ),
    lora_request=None,
)
```
