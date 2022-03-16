# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ctcdecode import CTCBeamDecoder
import torch
from torch import TensorType
from decoding_algorithms.ctc_decoder_base import CTCDecoderBase
from typing import Dict, List
from fairseq.data.ctc_dictionary import CTCDictionary


class CTCBeamSearchDecoder(CTCDecoderBase):
    """
    CTC Beam Search Decoder
    """

    def __init__(self, dictionary: CTCDictionary, decoder_parameters: Dict) -> None:
        super().__init__(dictionary, decoder_parameters)
        self.decoder = CTCBeamDecoder(dictionary.symbols, model_path=decoder_parameters["model_path"],
                                      alpha=decoder_parameters["alpha"], beta=decoder_parameters["beta"],
                                      cutoff_top_n=decoder_parameters["cutoff_top_n"],
                                      cutoff_prob=decoder_parameters["cutoff_prob"],
                                      beam_width=decoder_parameters["beam_width"],
                                      num_processes=decoder_parameters["num_processes"],
                                      blank_id=dictionary.blank(),
                                      log_probs_input=True)  # This is true since our criteria script returns log_prob.

    def decode(self, log_prob: TensorType, **kwargs) -> List[List[int]]:
        """
        Decoding function for the CTC beam search decoder.
        """
        if log_prob.dtype != torch.float16:
            log_prob = log_prob.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(log_prob)
        top_beam_tokens = beam_results[:, 0, :]  # extract the most probable beam
        top_beam_len = out_lens[:, 0]
        mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
            repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
        top_beam_tokens[~mask] = self.dictionary.pad()  # mask out nonsense index with pad index.
        top_beam_tokens = top_beam_tokens.cpu().tolist()
        for i in range(0, len(top_beam_tokens)):
            current_summary_index = top_beam_tokens[i]
            # Since ctc beam search decoder does post-process for us, we don't need to call post-processing again.
            if self.truncate:
                current_summary_index = current_summary_index[:self.desired_length]
            top_beam_tokens[i] = current_summary_index

        return top_beam_tokens
