# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from fairseq import utils
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models.transformer.transformer_config import TransformerEncoderConfig
from fairseq.models.transformer.transformer_base import Embedding
from fairseq.models.nat.fairseq_nat_model import FairseqNATEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class FairseqNATEncoderOnly(BaseFairseqModel):
    """
    This class gives an encoder-only model, where we directly project the encoder output (hidden states)
    to vocabulary.
    """
    def __init__(self, args, encoder):
        """
        Initialization of the class
        """
        super().__init__()
        self.encoder = encoder

        def output_projection(latent_space):
            return torch.matmul(latent_space, self.encoder.embed_tokens.weight.transpose(0, 1))

        self.output_projection = output_projection

    def forward(self, src_tokens, src_lengths):
        """
        Forward function of our customized encoder-only model
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        latent_space = encoder_out["encoder_out"][0].transpose(0, 1)
        projected_encoder_out = self.output_projection(latent_space)
        return projected_encoder_out

    def forward_encoder(self, src_tokens, src_lengths):
        """
        Return the encoder output
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        return encoder_out

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerEncoderConfig(), delete_default=True, with_prefix=""
        )
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )


    @classmethod
    def build_embedding(cls, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = FairseqNATEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """
        Instead of directly instancing the class through initialization, fairseq needs an extra
        class function build_model to build an instance of the given class.
        """
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            encoder_embed_tokens = cls.build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        else:
            raise ValueError("--share-all-embeddings must be True for this encoder-only model")
        # if cfg.offload_activations:
        #     cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        return cls(args, encoder)

    @property
    def has_encoder(self):
        """
        Property function which tells whether the model has an encoder
        """
        return True

    @property
    def has_decoder(self):
        """
        Property function which tells whether the model has a decoder
        """
        return False

    def get_normalized_probs_scriptable(self, net_output: Tensor, log_probs: bool, something_els=None):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        # syntactic sugar for simple models which don't have a decoder
        # (e.g., the classification tutorial)
        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
