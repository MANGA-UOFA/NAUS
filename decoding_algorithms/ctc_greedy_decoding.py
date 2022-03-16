# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from decoding_algorithms.ctc_decoder_base import CTCDecoderBase
from typing import Dict, List
from fairseq.data.ctc_dictionary import CTCDictionary
from torch import TensorType
import torch


class CTCGreedyDecoder(CTCDecoderBase):
    """
    CTC greedy decoding decoder
    """

    def __init__(self, dictionary: CTCDictionary, decoder_parameters: Dict) -> None:
        super().__init__(dictionary, decoder_parameters)

    def decode(self, output_logits: TensorType, **kwargs) -> List[List[int]]:
        """
        Decoding function of the CTC greedy Decoder.
        """
        if output_logits.dtype != torch.float16:
            output_logits = output_logits.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        _, tokens = output_logits.topk(1, -1)  # Use topk to get a fair comparison with the length control algorithm
        decoded_summary_list = []
        tokens = tokens.squeeze(-1).tolist()
        for sample in tokens:
            sample = self.ctc_post_processing(sample)
            if self.truncate and len(sample) > self.desired_length:
                sample = sample[:self.desired_length]
            decoded_summary_list.append(sample)
        return decoded_summary_list
