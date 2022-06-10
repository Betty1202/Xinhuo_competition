import TopsInference
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange
from .models import top_k_top_p_filtering


class GenerativeT5_DTU(torch.nn.Module):
    """ This wrapper utility function implements a single beam search to generate efficiently text.
        A lot of the credit goes to the huggingface team and its chief scientist Thomas Wolf whose implementation I based
        myself off.
    """

    def __init__(self, model, tokenizer, decoder_start_token_id=0, eos_token_id=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
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
        with TopsInference.device(0, 0):
            # load encoder and decoder
            self.model = TopsInference.load(self.model)

            new_tokens = torch.tensor(())
            new_logits = []
            encoder_idx = torch.tensor(self.tokenizer(prompt)['input_ids'])[:max_context_length - 1].unsqueeze(0)
            encoder_mask = torch.ones((1, len(encoder_idx[0])), dtype=torch.long)
            encoder_idx = F.pad(encoder_idx, (0, 256 - len(encoder_idx[0])), 'constant', 0).numpy().astype(np.int64)
            encoder_mask = F.pad(encoder_mask, (0, 256 - len(encoder_mask[0])), 'constant', 0).numpy().astype(np.int64)
            temperature = temperature

            repetition_penalty = repetition_penalty
            top_k = top_k
            top_p = top_p

            # The sequence now needs to start with a
            decoder_idx = self.decoder_start_token_id * torch.ones((1, 1), dtype=torch.long)
            decoder_mask = torch.ones((1, 1), dtype=torch.long)

            for _ in trange(max_length):
                outputs = []
                decoder_idx_input = F.pad(decoder_idx, (0, 256 - len(decoder_idx[0])), 'constant', 0).numpy().astype(
                    np.int64)
                decoder_mask_input = F.pad(decoder_mask, (0, 256 - len(decoder_mask[0])), 'constant', 0).numpy().astype(
                    np.int64)
                self.model.run(
                    [encoder_idx, encoder_mask, decoder_idx_input, decoder_mask_input], outputs,
                    TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                outputs = torch.tensor(outputs[0][0])
                next_token_logits = outputs[len(decoder_idx[0]) - 1, :] / (temperature if temperature > 0 else 1.0)
                if int(next_token_logits.argmax()) == self.eos_token_id:
                    break
                new_logits.append(next_token_logits)
                for _ in set(decoder_idx.view(-1).tolist()):
                    next_token_logits[_] /= repetition_penalty
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                else:
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                decoder_idx = torch.cat((decoder_idx, next_token.unsqueeze(0)), dim=1)
                decoder_mask = torch.cat((decoder_mask, torch.ones((1, 1), dtype=torch.long)), dim=1)
                new_tokens = torch.cat((new_tokens.float(), next_token.float()), 0)

        return self.tokenizer.decode(new_tokens.long()), new_logits
