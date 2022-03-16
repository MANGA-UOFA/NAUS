# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import numpy as np
import torch
from fairseq import utils
import time


class SummarizationCTCGenerator(object):
    """
    A generator designed for summarization task
    """
    def __init__(
        self,
        tgt_dict,
        models=None,
        retain_dropout=False,
        adaptive=True,
    ):
        f"""
        Generates summaries based on some decoding algorithm. 
        Notice this generator currently does not support parallel generation. 

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.blank = tgt_dict.blank()
        self.vocab_size = len(tgt_dict)
        self.retain_dropout = retain_dropout
        self.adaptive = adaptive
        self.models = models
        self.model_cfg = models[0].cfg

    def generate_batched_itr(
        self,
        data_itr,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual summarizations.

        Args:
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        start_time = time.time()
        net_output = model(sample)
        lprobs = model.get_normalized_probs(net_output, log_probs=True).contiguous()  # (T, B, C) from the encoder
        hyp_summaries = model.ctc_decoder.decode(lprobs, source_length=sample["net_input"]["src_lengths"])
        end_time = time.time()
        inference_time = end_time - start_time
        finalized_summaries = []
        for i in range(0, len(hyp_summaries)):
            # We also propagate the inference time back to the main function.
            finalized_summaries.append([{"tokens": hyp_summaries[i], "inference_time": inference_time, "target": sample["target"][i]}])

        return finalized_summaries
