#!/usr/bin/env python

if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    from datasets import load_dataset
    from sardalign.utils import dsu2pua

    from ssi.tokenizer import setup_llama3_tokenizer

    base = Path("/mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/original/tokenizer.model")
    ext = Path("/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus/original/tokenizer.model")
    tok_base, _ = setup_llama3_tokenizer(base, verbose=False)
    tok_ext, _ = setup_llama3_tokenizer(ext, verbose=False)

    # tokenizer a string containing DUSs - taken directly from
    dev = load_dataset("anilkeshwani/MLS_english_train_strat_sample_aligned_hubert", split="dev")
    sample = next(iter(dev))
    text = (
        "English Speech: "
        + "".join(dsu2pua(dsu) for dsu in sample["speech_tokens"])
        + "\nEnglish Text: "
        + sample["transcript"]
    )
    print(text)

    enc_base = tok_base.encode(text)
    enc_ext = tok_ext.encode(text)

    # print(enc_base)
    # print(enc_ext)

    enc_base_no_spec = [_ for _ in enc_base if _ not in tok_base.special_tokens.values()]
    enc_ext_no_spec = [_ for _ in enc_ext if _ not in tok_ext.special_tokens.values()]

    print(f"{enc_base_no_spec == enc_ext_no_spec = }")

    text_ss = (  # ss -> space-separated, as related to the DSUs portion of the text
        "English Speech: "
        + "".join(dsu2pua(dsu) for dsu in sample["speech_tokens"])
        + "\nEnglish Text: "
        + sample["transcript"]
    )

    enc_base_ss = tok_base.encode(text_ss)
    enc_ext_ss = tok_ext.encode(text_ss)

    enc_base_no_spec_ss = [_ for _ in enc_base if _ not in tok_base.special_tokens.values()]
    enc_ext_no_spec_ss = [_ for _ in enc_ext if _ not in tok_ext.special_tokens.values()]

    print(f"{enc_base_no_spec_ss == enc_ext_no_spec_ss = }")

    dsus = "".join(dsu2pua(dsu) for dsu in sample["speech_tokens"])

    print(tok_ext.tt_model.tt_model.encode(dsus))
    breakpoint()
    print([tok_ext.tt_model.tt_model.encode_single_token(dsu) for dsu in dsus])
    breakpoint()

    print(tok_base.tt_model.tt_model.encode(dsus))
    breakpoint()
    print([tok_base.tt_model.tt_model.encode_single_token(dsu) for dsu in dsus])
    breakpoint()
