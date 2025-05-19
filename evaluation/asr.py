from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from evaluate import load
from vllm import LLM, SamplingParams
from whisper_normalizer.basic import BasicTextNormalizer


# TODO Remove before committing
# covost_path="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/Continual_pretraining__covost_data/CoVoST_complete_test.json"
# LS_clean = "/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-clean.json"
# LS_other = "/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-other.json"

# {
#     "instruction": "English speech: <extra_id_38><extra_id_3814><extra_id_715>...<extra_id_1867> \n English text:",
#     "output": "And did you put the envelope in your pocket?",
# }
# {
#     "instruction": "English speech: <extra_id_38><extra_id_3814><extra_id_715><extra_id_3057><extra_id_941><extra_id_3363><extra_id_877><extra_id_3363><extra_id_1240><extra_id_877><extra_id_2964><extra_id_1167><extra_id_2654><extra_id_2964><extra_id_1167><extra_id_212><extra_id_2885><extra_id_204><extra_id_1922><extra_id_1714><extra_id_214><extra_id_1714><extra_id_774><extra_id_877><extra_id_2085><extra_id_4321><extra_id_531><extra_id_1249><extra_id_2754><extra_id_1406><extra_id_3610><extra_id_4249><extra_id_4108><extra_id_1634><extra_id_2389><extra_id_3495><extra_id_1537><extra_id_3120><extra_id_3665><extra_id_4843><extra_id_2858><extra_id_4312><extra_id_3161><extra_id_2753><extra_id_894><extra_id_3170><extra_id_452><extra_id_2466><extra_id_1918><extra_id_2428><extra_id_2712><extra_id_2566><extra_id_832><extra_id_1064><extra_id_100><extra_id_210><extra_id_1323><extra_id_4317><extra_id_3487><extra_id_4862><extra_id_1545><extra_id_1585><extra_id_2548><extra_id_4418><extra_id_3878><extra_id_1186><extra_id_3609><extra_id_3727><extra_id_1072><extra_id_3818><extra_id_4960><extra_id_227><extra_id_2206><extra_id_4892><extra_id_2462><extra_id_1761><extra_id_595><extra_id_628><extra_id_2419><extra_id_422><extra_id_1270><extra_id_1949><extra_id_366><extra_id_2485><extra_id_779><extra_id_4476><extra_id_172><extra_id_2753><extra_id_1210><extra_id_3170><extra_id_1082><extra_id_1663><extra_id_116><extra_id_3012><extra_id_59><extra_id_2547><extra_id_1743><extra_id_903><extra_id_2714><extra_id_215><extra_id_3817><extra_id_2869><extra_id_1480><extra_id_193><extra_id_1480><extra_id_819><extra_id_2254><extra_id_4710><extra_id_3071><extra_id_4948><extra_id_3071><extra_id_4948><extra_id_531><extra_id_1063><extra_id_3367><extra_id_1867> \n English text:",
#     "output": "And did you put the envelope in your pocket?",
# }


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str, help="Hugging Face checkpoint path")
    parser.add_argument("--tokenizer", type=Path, help="Path to SentencePiece .model tokenizer file")
    parser.add_argument("--test_jsonl", type=Path, help="Path to test set JSON lines manifest file")
    sampling_params_arg_group = parser.add_argument_group("SamplingParams")
    sampling_params_arg_group.add_argument("--temperature", type=float, default=0)
    sampling_params_arg_group.add_argument("--max_tokens", type=int, default=128)
    sampling_params_arg_group.add_argument(
        "--stop",
        type=str,
        nargs="+",
        default=[r"<\s>"],
        help="List of strings that stop the generation when they are generated. "
        "The returned output will not contain the stop strings.",
    )
    args = parser.parse_args()
    return args


def main(args: Namespace):
    text_normalizer = BasicTextNormalizer(remove_diacritics=False, split_letters=False)
    sampling_params = SamplingParams(
        stop=args.stop,
        max_tokens=args.max_tokens,  # 64
        temperature=args.temperature,
        # NOTE explicit defaults; may be useful test-time (hyper)parameters to tune
        frequency_penalty=0,
        repetition_penalty=1,
        top_k=-1,
        seed=None,
        use_beam_search=False,
        length_penalty=1,
        early_stopping=False,
    )

    # prepare into prompt list
    def read_data(data_path):
        prompts_df = pd.read_json(data_path)
        ref = prompts_df["output"].to_list()
        prompts = prompts_df["instruction"].to_list()
        return [prompts[i].strip(" ").strip("\n") for i in range(len(prompts))], [
            text_normalizer(ref[i]) for i in range(len(ref))
        ]

    def vllm_generate(out_file, model_path, prompts, ref):
        generated = []
        llm = LLM(model=model_path, tokenizer=tokenizer_path, tokenizer_mode="slow")
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated.append(text_normalizer(generated_text))

            out_file.write(text_normalizer(generated_text))
            out_file.write("\n")

        # compute metric
        wer = load("wer")

        results_wer = wer.compute(predictions=generated, references=ref)
        results = {"wer": results_wer}
        print(results)
        return results, generated

    for d in data.keys():
        f = open(out_root + "ASR_l7b_IT_{}.txt".format(d), "a")
        prompts, ref = read_data(data[d])
        results, generated = vllm_generate(f, model_path, prompts, ref)
