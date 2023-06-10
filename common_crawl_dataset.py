import json
import os
import random
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import LayoutLMv3Processor
from transformers.data.data_collator import DataCollatorMixin

from masking_generator import MaskingGenerator


def make_mlm_mask(length, batch_size, probability=0.115, lamda=3) -> torch.tensor:
    """

    Parameters
    ----------
    length: int
    batch_size: int
    probability: float
        Bernoulli distribution parameter to determine whether a token is the start of a masked span.
    lamda: int
        Poisson distribution parameter to determine the length of a masked span.

    The layoutLMv3 paper used lamda=3 and said that 30% of the tokens were masked. Empirically probability=0.115 and
    lamda=3 result in 30% of the tokens being masked.

    This function does not attempt to create non-overlapping spans. However the Bernoulli probability is typically low
    enough that this doesn't seem that significant.

    Returns
    -------
    torch.tensor
    """

    p = torch.full((batch_size, length), probability)

    # Generate a binary array from a Bernoulli distribution. Ones indicate positions where a masked span begins.
    b = torch.bernoulli(p)

    # For each element where a span begins sample the span length from a Poisson distribution
    span_length = torch.poisson(b * lamda)

    batch_idx, start_idx = torch.where(span_length)
    end_idx = (start_idx + span_length[batch_idx, start_idx]).long()

    mask = torch.zeros(batch_size, length).bool()
    for batch, start, end in zip(batch_idx, start_idx, end_idx):
        mask[batch, start:end] = True
    return mask


def generate_wpa_labels(bbox, upsampled_mim_mask, mlm_mask, ignore_label, overlap_threshold=0.98):
    # A description of how WPA labels are generated from the first author of the LayoutLMv3 paper
    # https://github.com/microsoft/unilm/issues/785
    batch_size, seq_length, _ = bbox.size()
    _, _, image_size, _ = upsampled_mim_mask.size()
    word_coords = (bbox * image_size / 1000).round().long()
    wpa_labels = torch.zeros((batch_size, seq_length), dtype=int)
    for example_idx in range(batch_size):
        for word_idx in range(seq_length):
            word_is_masked = mlm_mask[example_idx, word_idx]
            coords = word_coords[example_idx, word_idx]
            if word_is_masked or all(coords == 0):
                wpa_labels[example_idx, word_idx] = ignore_label
            else:
                left, top, right, bottom = coords.split(1)
                word_slice = upsampled_mim_mask[example_idx, left:right, top:bottom]
                if word_slice.sum() / word_slice.numel() > overlap_threshold:
                    wpa_labels[example_idx, word_idx] = 1
    return wpa_labels


@dataclass
class DataCollatorForLayoutPretraining(DataCollatorMixin):

    processor: LayoutLMv3Processor
    image_masking_generator: MaskingGenerator
    return_tensors: str = "pt"
    ignore_label: int = -100

    @staticmethod
    def make_example(doc_features: dict, page_num_str: str):
        image_bytes = doc_features[f'{page_num_str}.png']
        image = Image.open(BytesIO(image_bytes))
        width, height = image.size

        word_df = pd.read_json(BytesIO(doc_features[f'{page_num_str}.word']))
        words = word_df.text.tolist()

        # TODO use line boxes
        word_df.x = (1000 * word_df.x / width).round().astype(int)
        word_df.x2 = (1000 * word_df.x2 / width).round().astype(int)
        word_df.y = (1000 * word_df.y / height).round().astype(int)
        word_df.y2 = (1000 * word_df.y2 / height).round().astype(int)
        boxes = list(zip(word_df.x, word_df.y, word_df.x2, word_df.y2))

        # TODO make this not random
        image_tokens = np.random.randint(low=0, high=8192, size=(16, 16))

        return image, words, boxes, image_tokens

    def get_doc_examples(self, doc_features):
        page_nums = [key.split('.')[0]
                     for key in doc_features.keys()
                     if key.endswith('png')]
        if len(page_nums) == 1:
            # TODO is this OK? Is there an alternative?
            selected_page_nums = [page_nums[0], page_nums[0]]
        else:
            selected_page_nums = random.sample(page_nums, k=2)
        page_1, page_2 = selected_page_nums
        example_1 = self.make_example(doc_features, page_1)
        example_2 = self.make_example(doc_features, page_2)
        return example_1, example_2

    def torch_call(self, doc_features_list):
        import torch

        seq_length = self.processor.tokenizer.model_max_length
        # images, words, boxes, image_tokens = list(map(list, zip(*features)))

        images, words, boxes, image_tokens, doc_ids = [], [], [], [], []

        for doc_features in doc_features_list:
            doc_id = int(doc_features['__key__'])
            doc_ids.extend([doc_id, doc_id])

            doc_example_1, doc_example_2 = self.get_doc_examples(doc_features)
            images.extend([doc_example_1[0], doc_example_2[0]])
            words.extend([doc_example_1[1], doc_example_2[1]])
            boxes.extend([doc_example_1[2], doc_example_2[2]])
            image_tokens.extend([doc_example_1[3], doc_example_2[3]])
        image_tokens = np.array(image_tokens)

        batch = self.processor(images, words, boxes=boxes, padding='max_length',
                               truncation=True, return_tensors=self.return_tensors)
        batch_size, _, image_size, _ = batch.pixel_values.size()

        batch['doc_id'] = torch.LongTensor(doc_ids)

        # Generate a mask indicating which tokens to mask for MLM
        mlm_mask = make_mlm_mask(seq_length, batch_size)
        special_tokens_mask = torch.tensor([encoding.special_tokens_mask for encoding in batch.encodings])
        mlm_mask = (mlm_mask & ~special_tokens_mask).bool()

        # To create the MLM task labels clone the input ids and set non-masked ids to the ignore_label
        text_labels = batch.input_ids.clone()
        text_labels.masked_fill_(~mlm_mask, self.ignore_label)
        batch['text_labels'] = text_labels

        # Set the tokens to be masked for MLM to the <MASK> token
        batch.input_ids.masked_fill_(mlm_mask, self.processor.tokenizer.mask_token_id)

        # Generate a mask indicating which tokens to mask for MIM
        mim_mask = torch.tensor(np.array([self.image_masking_generator() for _ in range(batch_size)])).bool()

        # The mask is 16x16 (14x14 in the paper) but the image is 224x224 pixels so up-sample before applying the mask
        upsampled_mim_mask = torch.kron(mim_mask, torch.ones((14, 14))).unsqueeze(-1).bool()
        permuted_image = torch.permute(batch.pixel_values, (0, 2, 3, 1))
        masked_image = torch.permute(permuted_image * ~upsampled_mim_mask, (0, 3, 1, 2))
        batch.pixel_values = masked_image

        # TODO verify that -100 is the correct label for the class token and that it lines up with the class token in
        #  the image token sequence

        wpa_labels = generate_wpa_labels(batch.bbox, upsampled_mim_mask, mlm_mask, self.ignore_label)
        batch['wpa_labels'] = wpa_labels

        image_labels = torch.LongTensor(image_tokens)
        image_labels.masked_fill_(~mim_mask, self.ignore_label)
        image_labels = image_labels.view(batch_size, -1)
        cls_labels = torch.tensor([[self.ignore_label]] * batch_size)
        image_labels = torch.cat([cls_labels, image_labels], dim=1)
        batch['image_labels'] = image_labels

        return batch
