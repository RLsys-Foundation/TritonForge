from typing import Dict, List, Tuple

from transformers import AutoTokenizer


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type

        # Only get system message length for tokenizers that need it
        if tokenizer_type in ["qwen", "distill_qwen"]:
            self.system_message_length, self.gen_token_length = self.get_system_message_length()
        else:
            self.system_message_length = 0
            self.gen_token_length = 0

    def get_response_lengths(self, loss_masks: List[List[int]]) -> List[int]:
        return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> Tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(self, messages: List[Dict]) -> Tuple[List[int], List[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            message_ids = self.tokenizer.apply_chat_template([message], tokenize=True)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(self, messages: List[Dict]) -> Tuple[List[int], List[int]]:
        prompt = self.tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length
        return token_ids, loss_mask

    def gen_multi_turn_loss_mask_llama(self, messages: List[Dict]) -> Tuple[List[int], List[int]]:
        """Generate loss mask for Llama/KernelLLM models.

        This method handles Llama3.1-based models like KernelLLM by:
        1. Applying the chat template to get properly formatted tokens
        2. Creating masks that only train on assistant responses
        3. Supporting multi-turn conversations
        """
        all_token_ids = []
        all_loss_masks = []

        # Process all messages together to get proper formatting
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Now we need to identify which parts are assistant responses
        # We'll do this by tokenizing each message separately and finding them in the full sequence
        current_pos = 0

        for i, message in enumerate(messages):
            # Get the formatted version of just this message
            if i == 0:
                # First message includes any system prompt
                single_msg_text = self.tokenizer.apply_chat_template(
                    messages[: i + 1], tokenize=False, add_generation_prompt=(message["role"] != "assistant")
                )
            else:
                # For subsequent messages, get the incremental part
                prev_text = self.tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True
                )
                curr_text = self.tokenizer.apply_chat_template(
                    messages[: i + 1], tokenize=False, add_generation_prompt=(message["role"] != "assistant")
                )
                single_msg_text = curr_text[len(prev_text) :]

            single_msg_tokens = self.tokenizer(single_msg_text, add_special_tokens=False)["input_ids"]
            msg_len = len(single_msg_tokens)

            # Create mask: 1 for assistant responses, 0 for everything else
            if message["role"] == "assistant":
                # For assistant messages, we want to train on the response
                # But not on any special tokens or formatting
                loss_mask = [1] * msg_len
            else:
                loss_mask = [0] * msg_len

            all_loss_masks.extend(loss_mask)
            current_pos += msg_len

        # Ensure we have the right length
        if len(all_loss_masks) < len(full_tokens):
            # Pad with zeros if needed (for any trailing tokens)
            all_loss_masks.extend([0] * (len(full_tokens) - len(all_loss_masks)))
        elif len(all_loss_masks) > len(full_tokens):
            # Truncate if too long
            all_loss_masks = all_loss_masks[: len(full_tokens)]

        return full_tokens, all_loss_masks

    def get_loss_mask(self, messages: List[Dict]) -> List[int]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages)

            return self.gen_multi_turn_loss_mask_qwen(messages)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages)
        elif self.tokenizer_type in ["llama", "kernelllm", "llama3", "llama3.1"]:
            return self.gen_multi_turn_loss_mask_llama(messages)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_text_from_loss_mask(self, token_ids: List[int], loss_masks: List[int]) -> List[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
