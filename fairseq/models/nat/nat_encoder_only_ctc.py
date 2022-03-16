# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from distutils.util import strtobool

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.fairseq_nat_encoder_only_model import FairseqNATEncoderOnly
import logging
import torch

try:
    # Decoder for naive ctc beam search, not runnable on Compute Canada
    from decoding_algorithms.ctc_beam_search import CTCBeamSearchDecoder
except:
    CTCBeamSearchDecoder = None
    print("Failed to load CTCBeamSearchDecoder, ignore this message if you are using Compute Canada. \n")
# Decoders for word-level length control
from decoding_algorithms.ctc_greedy_decoding import CTCGreedyDecoder
from decoding_algorithms.ctc_scope_search_length_control import CTCScopeSearchLengthControlDecoder

# Decoders for char length control


logger = logging.getLogger(__name__)


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("nat_encoder_only_ctc")
class NATransformerEncoderOnlyModel(FairseqNATEncoderOnly):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.ctc_decoder = self.create_ctc_decoder(args)  # This is not Transformer decoder!
        self.cfg = args

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser, **kwargs):
        FairseqNATEncoderOnly.add_args(parser)
        parser.add_argument(
            '--decoding_algorithm',
            default="ctc_greedy_decoding",
            type=str,
            choices=["ctc_greedy_decoding", "ctc_beam_search", "ctc_beam_search_length_control"],

            help="Options to control the the CTC decoding method",
        )
        parser.add_argument(
            '--truncate_summary',
            default=False,
            type=strtobool,
            help="Whether to truncate the generated summaries. Notice this is only valid for ctc_greedy decoding "
                 "and ctc_beam_search",
        )
        parser.add_argument(
            '--force_length',
            default=False,
            type=strtobool
        )
        parser.add_argument(
            '--desired_length',
            default=10,
            type=int
        )
        parser.add_argument(
            '--use_length_ratio',
            default=False,
            type=strtobool
        )
        parser.add_argument(
            '--k',
            default=10,
            type=int
        )
        parser.add_argument(
            '--beam_size',
            default=6,
            type=int
        )
        parser.add_argument(
            '--marg_criteria',
            default="max",
            type=str
        )
        parser.add_argument(
            '--scope',
            default=3,
            type=int
        )
        parser.add_argument(
            '--scaling_factor',
            default=4,
            type=int
        )

    def decode_lprob_to_token_index(self, lprobs, sample=None):
        if sample is None:
            source_length = None
        else:
            source_length = sample["net_input"]["src_lengths"]
        return self.ctc_decoder.decode(lprobs, source_length=source_length)

    def create_ctc_decoder(self, args):
        """
        Create a CTC decoder to map logits to words, based on the user-specified decoding choice
        """
        decoding_algorithm = getattr(args, 'decoding_algorithm')
        if decoding_algorithm == "ctc_greedy_decoding":
            assert getattr(args, 'force_length') == False, "Cannot force length for greedy decoding"
            decoding_params = {"truncate_summary": getattr(args, 'truncate_summary'),
                               "desired_length": getattr(args, 'desired_length'), }
            decoder = CTCGreedyDecoder(self.encoder.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_beam_search":
            assert getattr(args, 'force_length') == False, "Cannot force length for ctc naive beam search"
            decoding_params = {
                "truncate_summary": getattr(args, 'truncate_summary'),
                "desired_length": getattr(args, 'desired_length'),
                "model_path": None,
                "alpha": 0,
                "beta": 0,
                "cutoff_top_n": getattr(args, 'k'),
                "cutoff_prob": 1.0,
                "beam_width": getattr(args, 'beam_size'),
                "num_processes": 4,
                "log_probs_input": True}
            decoder = CTCBeamSearchDecoder(self.encoder.dictionary, decoding_params)
            print("Successfully created the CTC Beam search decoder")
        elif decoding_algorithm == "ctc_beam_search_length_control":
            assert getattr(args, 'truncate_summary') == False, "Cannot truncate summary with length control decoding"
            decoding_params = {
                'force_length': getattr(args, 'force_length'),
                "desired_length": getattr(args, 'desired_length'),
                "use_length_ratio": getattr(args, 'use_length_ratio'),
                "k": getattr(args, 'beam_size'),
                "beam_size": getattr(args, 'beam_size'),
                "scope": getattr(args, 'scope'),
                "marg_criteria": getattr(args, 'marg_criteria', 'max'),
                # truncate_summary is a dummy variable since length control does not need it
                "truncate_summary": getattr(args, 'truncate_summary')
            }
            decoder = CTCScopeSearchLengthControlDecoder(self.encoder.dictionary, decoding_params)
        else:
            raise (NotImplementedError, "%s is not supported" % decoding_algorithm)
        return decoder

    def forward(self, sample, something_else=None):
        """
        Forward function of the model
        """
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        projected_encoder_out = super().forward(src_tokens, src_lengths)

        return projected_encoder_out

    def initialize_output_tokens_by_src_tokens(self, src_tokens):
        if not self.copy_src_token:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            if self.args.src_upsample_scale > 2:
                length_tgt = length_tgt * self.args.src_upsample_scale
            else:
                length_tgt = length_tgt * self.args.src_upsample_scale  # + 10
            max_length = length_tgt.clamp_(min=2).max()
            idx_length = utils.new_arange(src_tokens, max_length)

            initial_output_tokens = src_tokens.new_zeros(
                src_tokens.size(0), max_length
            ).fill_(self.pad)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] < length_tgt[:, None], self.unk
            )
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
            return initial_output_tokens
        else:
            if self.args.src_upsample_scale <= 1:
                return src_tokens

            def _us(x, s):
                B = x.size(0)
                _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
                return _x

            return _us(src_tokens, self.args.src_upsample_scale)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output
            if log_probs:
                return logits.log_softmax(dim=-1)
            else:
                return logits.softmax(dim=-1)


@register_model_architecture(
    "nat_encoder_only_ctc", "nat_encoder_only_ctc"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
