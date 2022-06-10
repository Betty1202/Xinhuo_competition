import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange


class CombinedDecoder(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    def forward(self, input_ids, encoder_hidden_states):
        decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)[0] * \
                         (self.config.d_model ** -0.5)
        return self.lm_head(decoder_output)


class SimplifiedT5Encoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Function created by Thomas Wolf of the huggingface team
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
            logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class GenerativeT5(torch.nn.Module):
    """ This wrapper utility function implements a single beam search to generate efficiently text.
        A lot of the credit goes to the huggingface team and its chief scientist Thomas Wolf whose implementation I based
        myself off.

    """

    def __init__(self, model, tokenizer, onnx=False, cuda=False, decoder_start_token_id=0, eos_token_id=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.onnx = onnx
        self.cuda = cuda
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

    def forward(self, prompt, max_length, temperature=1., repetition_penalty=1., top_k=50, top_p=0,
                max_context_length=512):
        """ Forward function to generate text after a prompt
            Args:
                prompt: str to run (don't forget to add at the beginning the task to run such as "summarize:"
                        or "translate English to German:"
                max_context_length: maximum number of tokens to use as context

        """
        with torch.no_grad():
            new_tokens = torch.tensor(())
            new_logits = []
            encoder_idx = torch.tensor(self.tokenizer(prompt)['input_ids'])[:max_context_length - 1].unsqueeze(0)
            if self.cuda and not self.onnx:
                encoder_idx = encoder_idx.cuda()
            temperature = temperature

            repetition_penalty = repetition_penalty
            top_k = top_k
            top_p = top_p

            # The sequence now needs to start with a
            generated = self.decoder_start_token_id * torch.ones((1, 1), dtype=torch.long)
            if self.cuda and not self.onnx:
                generated = generated.cuda()

            for _ in trange(max_length):
                if self.onnx:
                    outputs = torch.tensor(
                        self.model.run(None, {"input_ids": encoder_idx.cpu().numpy(),
                                              "attention_mask": np.ones(encoder_idx.shape, dtype=np.int),
                                              "decoder_input_ids": generated.cpu().numpy(),
                                              "decoder_attention_mask": np.ones(generated.shape, dtype=np.int)})[0][
                            0])
                else:
                    outputs = self.model(input_ids=encoder_idx,
                                         decoder_input_ids=generated)[0][0]
                next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)
                if int(next_token_logits.argmax()) == self.eos_token_id:
                    break
                new_logits.append(next_token_logits)
                for _ in set(generated.view(-1).tolist()):
                    next_token_logits[_] /= repetition_penalty
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                else:
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                new_tokens = torch.cat((new_tokens.float(), next_token.float()), 0)

        return self.tokenizer.decode(new_tokens.long()), new_logits
