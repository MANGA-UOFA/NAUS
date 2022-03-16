# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from decoding_algorithms.ctc_decoder_base import CTCDecoderBase


class CTCScopeSearchLengthControlDecoder(CTCDecoderBase):
    """
    An equivalent implementation of the CTC Beam Search with Length Control (BSLC).
    Comparing with naive BSLC, this implementation can generalize toward brute force search by increasing the scope.
    """

    def __init__(self, dictionary, decoder_parameters):
        super().__init__(dictionary, decoder_parameters)
        # Sample temporary variable
        self.sample_desired_length = None
        self.id_to_index_dict_list = None
        self.index_to_id_dict_list = None
        self.ctc_sequence_length = None
        self.prob_sequence = None
        self.prev_max_row_index = None
        self.scope_lprob_table = None
        self.transition_tracker_table = None
        self.single_prefix_index_table = None
        # Decoder configuration parameters
        self.force_length = decoder_parameters["force_length"]
        self.use_length_ratio = decoder_parameters["use_length_ratio"]
        self.k = decoder_parameters["k"]  # dummy variable
        self.beam_size = decoder_parameters["beam_size"]
        self.scope = decoder_parameters["scope"]
        self.margin_criteria = decoder_parameters["marg_criteria"]
        self.blank_index = dictionary.blank()
        self.replacing_value = float("-inf")
        self.device = None
        self.dtype = None
        # Assertions on decoder parameters
        assert self.scope > 1, "The scope must be positive integer"
        assert self.beam_size > 0, "Beam size are required to be positive"
        assert self.desired_length > 0, "The desired length should be greater than 0"
        # assert self.beam_size % 2 == 0, "The beam size must be even number"
        # Initialize reusable variables
        self.special_element_tuple_list = [list(range(self.k)), [0] * self.k]

    def top_k_filtering(self, column, lprob):
        """
            Get the top-k most probable token and their corresponding probabilities
            logits (tensor): the logits returned by the model at the current time step

            Return:
                values (tensor of size k): the probability of the most probable tokens, with the first one set to be the blank token
                index_to_id_dict (dict of size k): a dictionary mapping from the element index of the values vector to their real id
                repeated_element_index_list (list): a list of the index (row, column) indicating the repeating index
        """
        top_k_id_tensor = torch.zeros(self.k, dtype=torch.long, device=self.device)  # word id
        top_k_lprob_tensor = torch.zeros(self.k, dtype=self.dtype, device=self.device)
        # Record the blank token id, no matter whether it's in top k
        top_k_id_tensor[0] = self.blank_index
        top_k_lprob_tensor[0] = lprob[self.blank_index]
        # Find the k most probable words and their indexes
        naive_top_k_lprob, naive_top_k_id = lprob.topk(self.k)
        # Fill in the remaining slot of the top_k_id_tensor and top_k_lprob_tensor
        top_k_id_tensor[1:] = naive_top_k_id[naive_top_k_id != self.blank_index][:self.k - 1]
        top_k_lprob_tensor[1:] = naive_top_k_lprob[naive_top_k_id != self.blank_index][:self.k - 1]

        # create dictionaries mapping between index and ids
        index_to_id_dict = {k: v.item() for k, v in enumerate(top_k_id_tensor)}
        id_to_index_dict = {v.item(): k for k, v in enumerate(top_k_id_tensor)}

        # Record the dictionary
        self.index_to_id_dict_list[column] = index_to_id_dict
        self.id_to_index_dict_list[column] = id_to_index_dict

        if column == 0:
            # For the first column, there is no repeated element
            repeated_or_special_element_index_list = self.special_element_tuple_list
        else:
            prev_id_to_index_dict = self.id_to_index_dict_list[column - 1]
            prev_index_to_id_dict = self.index_to_id_dict_list[column - 1]
            # Find the overlapping words except blank token in the current top_k words and previous top_k words
            repeated_element_list = set(prev_id_to_index_dict.keys()).intersection(set(top_k_id_tensor[1:].tolist()))
            repeated_element_tuple_index_list = [[prev_id_to_index_dict[element] for element in repeated_element_list],
                                                 [id_to_index_dict[element] for element in repeated_element_list]]
            repeated_or_special_element_index_list = repeated_element_tuple_index_list
            repeated_or_special_element_index_list[0] += self.special_element_tuple_list[0]
            repeated_or_special_element_index_list[1] += self.special_element_tuple_list[1]

        repeated_or_special_element_index_list[0] = tuple(repeated_or_special_element_index_list[0])
        repeated_or_special_element_index_list[1] = tuple(repeated_or_special_element_index_list[1])

        return top_k_lprob_tensor, repeated_or_special_element_index_list

    def get_blank_token_prob(self, current_filtered_log_prob):
        only_special_cloned_prob = current_filtered_log_prob.clone()
        only_special_cloned_prob[1:] = self.replacing_value
        return only_special_cloned_prob

    def get_non_blank_token_prob(self, current_filtered_log_prob):
        only_non_special_cloned_prob = current_filtered_log_prob.clone()
        only_non_special_cloned_prob[0] = self.replacing_value
        return only_non_special_cloned_prob

    def scope_search_row_inference(self, column, reshaping_index, only_blank_prob, only_non_blank_prob,
                                   blank_or_repeated_transition_matrix, non_blank_non_repeated_transition_matrix,
                                   repeated_or_blank_transition_mask):
        """
        Perform actual table filling for each rows.
        """
        repeated_or_blank_transition_mask = repeated_or_blank_transition_mask[reshaping_index]
        repeated_or_blank_transition_mask = \
            repeated_or_blank_transition_mask.repeat(
                2 * [1] + (repeated_or_blank_transition_mask.dim() - 2) * [self.beam_size])
        # Initialization
        if column == 0:
            # Add the blank token probability to all dimensions of the first row of scope table
            self.scope_lprob_table[0] += only_blank_prob[reshaping_index]
            # Add the non-blank token probability to all dimensions of the second row of the scope table
            self.scope_lprob_table[1] += only_non_blank_prob[reshaping_index]
            # No other operation is needed for the first column
            return
        # Recursion
        # We first update the expansion row, then middle rows and the first row to avoid undesired in-place update
        # Expansion row
        if column + 1 < self.sample_desired_length:
            # Calculate the probability of each word in the pure transition.
            self.scope_lprob_table[column + 1] = self.scope_lprob_table[self.prev_max_row_index] \
                                                 + non_blank_non_repeated_transition_matrix[reshaping_index]
            # Record the previous row of these transitions.
            self.transition_tracker_table[column + 1, ..., ~repeated_or_blank_transition_mask, :] = \
                self.transition_tracker_table[self.prev_max_row_index, ..., ~repeated_or_blank_transition_mask, :]
            self.transition_tracker_table[column + 1, ..., ~repeated_or_blank_transition_mask, [column - 1]] \
                = self.prev_max_row_index

        # Middle rows
        # Transition probability from diagonal neighbours
        diagonal_transition_lprob \
            = self.scope_lprob_table[0:self.prev_max_row_index] + non_blank_non_repeated_transition_matrix[
            reshaping_index]
        # Transition probability from row neighbours
        row_transition_lprob \
            = self.scope_lprob_table[1:self.prev_max_row_index + 1] + blank_or_repeated_transition_matrix[
            reshaping_index]
        # Combine the two types of transitions into one Tensor
        sum_transition_lprob = diagonal_transition_lprob
        sum_transition_lprob[..., repeated_or_blank_transition_mask] \
            = row_transition_lprob[..., repeated_or_blank_transition_mask]
        self.scope_lprob_table[1:self.prev_max_row_index + 1] = sum_transition_lprob
        # Record the previous rows for each of the current row
        current_tracker_dim = self.transition_tracker_table[1:self.prev_max_row_index + 1,
                              ..., ~repeated_or_blank_transition_mask, [column - 1]].dim() - 1
        # Copy the transition history
        diagonal_transition_history = self.transition_tracker_table[0:self.prev_max_row_index].clone()
        diagonal_transition_history[..., ~repeated_or_blank_transition_mask, [column - 1]] += \
            torch.arange(0, self.prev_max_row_index, device=self.device)[(...,) + (None,) * current_tracker_dim] + 1
        row_transition_history = self.transition_tracker_table[1:self.prev_max_row_index + 1].clone()
        row_transition_history[..., repeated_or_blank_transition_mask, [column - 1]] += \
            torch.arange(1, self.prev_max_row_index + 1, device=self.device)[(...,) + (None,) * current_tracker_dim] + 1

        # Record the current chosen index
        self.transition_tracker_table[1:self.prev_max_row_index + 1, ..., ~repeated_or_blank_transition_mask, :] \
            = diagonal_transition_history[..., ~repeated_or_blank_transition_mask, :]
        self.transition_tracker_table[1:self.prev_max_row_index + 1, ..., repeated_or_blank_transition_mask, :] \
            = row_transition_history[..., repeated_or_blank_transition_mask, :]

        # First row
        # Add the blank token probability to all dimensions of the first row of scope table
        self.scope_lprob_table[0] = self.scope_lprob_table[0] + only_blank_prob[reshaping_index]
        # Set the previous row of the current first row to 0
        transition_index = torch.zeros(self.beam_size, device=self.device, dtype=torch.int8) - 1
        transition_index[0] = 1  # add 1 to encounter -1 in the initialization of the transition table
        self.transition_tracker_table[0, ..., column - 1] += transition_index[reshaping_index]

    def scope_search_column_inference(self, column):
        """
        Perform table (prob table and prefix table) filling for a single column
        """
        # Calculate some temporal variables
        remaining_scope = self.scope - column  # Remaining dimensions before filling in the current column
        self.prev_max_row_index = min(column, self.sample_desired_length - 1)  # Notice it's an index
        # Find the top_k probability
        current_step_lprob = self.prob_sequence[column].log()
        filtered_lprob, repeated_or_special_element_index_list = self.top_k_filtering(column, current_step_lprob)
        only_blank_cloned_prob = self.get_blank_token_prob(filtered_lprob)
        only_non_blank_cloned_prob = self.get_non_blank_token_prob(filtered_lprob)
        # Create a mask for non_blank_non_repeat_transitions
        repeated_or_blank_transition_mask = torch.zeros([self.k, self.k], dtype=torch.bool, device=self.device)
        repeated_or_blank_transition_mask[repeated_or_special_element_index_list] = True
        # Mask out the blank and repeated transitions
        non_blank_non_repeated_transition_matrix = filtered_lprob.expand(self.k, self.k).clone()
        non_blank_non_repeated_transition_matrix[repeated_or_blank_transition_mask] = self.replacing_value
        # Mask out the non-blank and non-repeated transitions
        blank_or_repeated_transition_matrix = filtered_lprob.expand(self.k, self.k).clone()
        blank_or_repeated_transition_matrix[~repeated_or_blank_transition_mask] = self.replacing_value

        # Find the appropriate reshaping index and marginalize the lprob matrix if needed.
        if remaining_scope > 0:
            reshaping_index = tuple((...,) + (None,) * (remaining_scope - 1))
        else:
            most_probable_word_index = self.margin_over_prob_table()
            reshaping_index = tuple((...,) + (None,) * (1 - 1))
            self.single_prefix_index_table[0:self.prev_max_row_index + 1, column - 1] = most_probable_word_index

        self.scope_search_row_inference(column, reshaping_index, only_blank_cloned_prob,
                                        only_non_blank_cloned_prob, blank_or_repeated_transition_matrix,
                                        non_blank_non_repeated_transition_matrix, repeated_or_blank_transition_mask)

    def margin_over_prob_table(self):
        """
        Marginalize over the first dimension of the table
        """
        remaining_axis = tuple(range(2, self.scope + 1))  # A tuple of the remaining axis after marginalization
        if self.margin_criteria == "mean":
            sum_old_prob_along_remaining_axis = torch.logsumexp(self.scope_lprob_table[:self.prev_max_row_index + 1],
                                                                dim=remaining_axis)
            most_probable_word_index = torch.argmax(sum_old_prob_along_remaining_axis, dim=1)
        elif self.margin_criteria == "filtered_mean":
            # Select token based on its average non-inf probability
            sum_lprob_along_remaining_axis \
                = torch.logsumexp(self.scope_lprob_table[:self.prev_max_row_index + 1], dim=remaining_axis)
            non_inf_sum = (self.scope_lprob_table[:self.prev_max_row_index + 1] != float("-inf")).long().sum(
                remaining_axis)
            sum_lprob_along_remaining_axis -= non_inf_sum.log()  # take the average
            sum_lprob_along_remaining_axis = torch.nan_to_num(sum_lprob_along_remaining_axis, float("-inf"))
            most_probable_word_index = torch.argmax(sum_lprob_along_remaining_axis, dim=1)
        elif self.margin_criteria == "max":
            # If we are using max as the select criteria, we select the token in the first axis that can lead to the
            # sub-sequence with the maximum probability
            max_old_prob_along_remaining_axis = torch.amax(self.scope_lprob_table[:self.prev_max_row_index + 1],
                                                           dim=remaining_axis)
            most_probable_word_index = torch.argmax(max_old_prob_along_remaining_axis, dim=1)
        else:
            raise NotImplementedError("Haven't designed other evaluation criteria")

        # Marginalize the lprob scope table based on the chosen words.
        repeat_index = tuple([1] + (self.scope - 1) * [1] + [self.beam_size])
        row_axis = torch.arange(0, self.prev_max_row_index + 1)
        self.scope_lprob_table[0:self.prev_max_row_index + 1] \
            = self.scope_lprob_table[row_axis, most_probable_word_index].unsqueeze(-1).repeat(repeat_index)

        # Marginalize the transition scope table based on the chosen words.
        repeat_index = tuple([1] + (self.scope - 1) * [1] + [self.beam_size] + [1])
        self.transition_tracker_table[0:self.prev_max_row_index + 1] \
            = self.transition_tracker_table[row_axis, most_probable_word_index].unsqueeze(-2).repeat(repeat_index)

        return most_probable_word_index

    def ctc_scope_search_length_control_initialization(self, logits):
        """
        Initialize some temporary variables
        """
        self.prob_sequence = logits.softmax(dim=-1)  # Get the log probability from logits
        self.ctc_sequence_length = len(self.prob_sequence)  # The length of the ctc output sequence.
        assert self.scope < self.ctc_sequence_length, \
            "The scope to reduce cannot exceed the length of the ctc output sequence"
        # This assertion results from our int8 declaration of transition tracking tensor
        assert self.ctc_sequence_length <= 2 ** 7, "The sequence length cannot exceed 128"
        # Track the probability of all transitions within a scope
        scope_lprob_dimensions = [self.sample_desired_length] + self.scope * [self.beam_size]
        self.scope_lprob_table = torch.zeros(scope_lprob_dimensions, dtype=self.dtype, device=self.device)
        # Track the parent rows of all choices
        transition_tracker_dimensions = \
            [self.sample_desired_length] + self.scope * [self.beam_size] + [self.ctc_sequence_length - 1]
        self.transition_tracker_table \
            = torch.zeros(transition_tracker_dimensions, dtype=torch.int8, device=self.device) - 1
        # This table stores the i-(scope-1) th prefix at the i-th column.
        self.single_prefix_index_table = \
            torch.zeros([self.sample_desired_length, self.ctc_sequence_length], dtype=torch.long,
                        device=self.device) - 1
        # Initialize a list of dictionary to record the mapping between index and word id at different time step.
        self.index_to_id_dict_list = [-1] * self.ctc_sequence_length
        self.id_to_index_dict_list = [-1] * self.ctc_sequence_length

    def determine_whether_length_control_needed(self, logits, source_length):
        """
        Determines whether length control is needed.
        """
        # Determine the desired summary length
        if self.use_length_ratio:
            # If the desired length is proportional to the input length, set desired length based on the input
            self.sample_desired_length = int(np.floor(0.01 * self.desired_length * source_length))
        else:
            # Otherwise, use the given length
            self.sample_desired_length = self.desired_length + 1  # Handle 0-length summary

        # Check whether the greedy decoding gives over-length summary
        _, greedy_summary_index = logits.max(-1)
        greedy_summary = self.ctc_post_processing(greedy_summary_index.tolist())
        # Determine whether we should adopt greedy decoding for this summary generation
        use_shorter_summary = (len(greedy_summary) <= self.sample_desired_length - 1) and (not self.force_length)
        source_too_short = (len(logits) <= self.sample_desired_length - 1)  # If source is shorter than desired length
        if use_shorter_summary or source_too_short:
            return False, greedy_summary
        else:
            return True, greedy_summary

    def finalize_generation(self):
        """
        Finalize a generation of word index
        """
        # We find the path of the last cell in the table and decode the path into a summary.
        # Fetch the trajectory
        maximum_index = (self.scope_lprob_table[-1] == torch.amax(self.scope_lprob_table[-1])).nonzero()[0].tolist()
        maximum_trajectory = self.transition_tracker_table[-1][tuple(maximum_index)]
        # Map the trajectory to summary prefix
        generated_summary_index = []
        for i in range(self.scope - 1, self.ctc_sequence_length - 1):
            current_row = maximum_trajectory[i]
            current_chosen_token_index = self.single_prefix_index_table[current_row, i]
            generated_summary_index.append(current_chosen_token_index.item())
        # Combine the postfix with the prefix
        generated_summary_index += maximum_index
        # Decode the index to word id
        generated_summary = []
        for i in range(0, self.ctc_sequence_length):
            current_token_index = generated_summary_index[i]
            current_column_dict = self.index_to_id_dict_list[i]
            current_token_id = current_column_dict[current_token_index]
            generated_summary.append(current_token_id)
        return generated_summary

    def ctc_scope_search_length_control(self, logits, source_length):
        """
        This function perform length control on the output CTC logits of the model decoder.
        """
        # First check whether length control is needed
        need_length_control, greedy_summary = self.determine_whether_length_control_needed(logits, source_length)
        if not need_length_control:
            # If not needed, return greedily decoded summary.
            return greedy_summary
        # Initialization of temporal variables
        self.ctc_scope_search_length_control_initialization(logits)
        # Main Loop
        for column in range(0, self.ctc_sequence_length):
            self.scope_search_column_inference(column)
        # Finalize a generation based on previously calculated variables
        generated_summary = self.finalize_generation()
        # Remove consecutively repeating tokens and blank tokens
        generated_summary = self.ctc_post_processing(generated_summary)
        assert len(generated_summary) == self.sample_desired_length - 1, "Generated summary has a wrong length"
        return generated_summary

    def decode(self, output_logits, source_length):
        """
        Decoding function for the CTC scope search decoder.
        """
        if output_logits.dtype != torch.float16:
            output_logits = output_logits.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        self.dtype = output_logits.dtype
        self.device = output_logits.device
        decoded_summary_list = []
        for i in range(0, len(output_logits)):
            decoded_summary = self.ctc_scope_search_length_control(output_logits[i], source_length)
            decoded_summary_list.append(decoded_summary)
        return decoded_summary_list
