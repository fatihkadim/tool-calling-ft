"""Completion-only data collator implementation.

TRL 1.x surumlerinde kaldirilan/degistirilen DataCollatorForCompletionOnlyLM
yerine, ChatML ve tool-calling formatina ozel, harici bagimliliklardan bagimsiz,
guvenilir ve dinamik padding destekli collator implementasyonu.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCompletionOnlyLM:
    """Prompt (system, tool schemas, user) tokenlarini -100 ile maskeleyip yalnizca

    assistant yanitina loss hesaplanmasini saglayan Data Collator.

    Args:
        response_template: Assistant yanitinin baslangicini belirten metin veya token ID listesi.
            Ornek: "<|im_start|>assistant\\n"
        tokenizer: Tokenizer ornegi (PreTrainedTokenizerBase)
        instruction_template: Coklu tur (multi-turn) konusmalarda kullanici mesajinin
            baslangicini belirten opsiyonel sablon.
        mlm: Masked LM bayragi (her zaman False olmali).
        ignore_index: PyTorch CrossEntropyLoss icin maskeleme degeri (varsayilan: -100).
    """

    response_template: Union[str, List[int]]
    tokenizer: PreTrainedTokenizerBase
    instruction_template: Optional[Union[str, List[int]]] = None
    mlm: bool = False
    ignore_index: int = -100

    def __post_init__(self):
        if isinstance(self.response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = list(self.response_template)

        if self.instruction_template is not None:
            if isinstance(self.instruction_template, str):
                self.instruction_token_ids = self.tokenizer.encode(
                    self.instruction_template, add_special_tokens=False
                )
            else:
                self.instruction_token_ids = list(self.instruction_template)
        else:
            self.instruction_token_ids = None

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """Batch orneklerini dinamik olarak pad eder ve assistant-only labels uretir."""
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i in range(len(examples)):
            input_ids = batch["input_ids"][i].tolist()
            # Varsayilan olarak tum tokenlari maskele
            labels[i, :] = self.ignore_index

            response_len = len(self.response_token_ids)
            idx = 0
            found_response = False

            while idx <= len(input_ids) - response_len:
                if input_ids[idx : idx + response_len] == self.response_token_ids:
                    found_response = True
                    response_start = idx + response_len
                    response_end = len(input_ids)

                    if self.instruction_token_ids is not None:
                        inst_len = len(self.instruction_token_ids)
                        for j in range(response_start, len(input_ids) - inst_len + 1):
                            if input_ids[j : j + inst_len] == self.instruction_token_ids:
                                response_end = j
                                break

                    # Assistant tokenlarini ac (pad token haric)
                    for k in range(response_start, response_end):
                        if (
                            self.tokenizer.pad_token_id is not None
                            and input_ids[k] == self.tokenizer.pad_token_id
                        ):
                            continue
                        labels[i, k] = input_ids[k]

                    idx = response_end
                else:
                    idx += 1

            # Eger response_template hic bulunamadiysa (orn. asiri truncation), tum sequence maskeli kalir
            if not found_response:
                labels[i, :] = self.ignore_index

        batch["labels"] = labels
        return batch

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        return self.torch_call(examples)
