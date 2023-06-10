import json
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper, FSSpecFileOpener, \
    SampleMultiplexer

from transformers import AutoProcessor

from common_crawl_dataset import make_mlm_mask, generate_wpa_labels
from masking_generator import MaskingGenerator

IGNORE_LABEL = -100

path_to_private_key = "/Users/alkymi/Downloads/alkymi-ds-211f0321d5a3.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_private_key
# url = 'gs://common-crawl-33-pdf-grouped-english'


def sample_page(example):
    page_ids = json.loads(example['.page_ids'])
    page_id = random.choice(page_ids)
    result = {
        '__key__': example['__key__'],
        '__page_id__': page_id,
        'png': Image.open(BytesIO(example[f'.{page_id}.png'])),
        'word_df': pd.read_json(BytesIO(example[f'.{page_id}.word'])),
        'image_tokens': json.loads(example[f'.{page_id}.image_tokens'])}
    return result


def process_batch(examples):
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base",
                                              apply_ocr=False)
    images, words, boxes, image_tokens, doc_ids = [], [], [], [], []

    for example in examples:
        image = example['png']
        images.append(image)

        words.append(example['word_df'].text.tolist())

        _boxes = example['word_df'][['x', 'y', 'x2', 'y2']].to_numpy()
        boxes.append(_boxes)

        _image_tokens = np.array(example['image_tokens']).reshape((16, 16))
        image_tokens.append(_image_tokens)

        doc_ids.append(int(example['__key__'].split('/')[-1]))

    image_tokens = np.array(image_tokens)

    batch = processor(images, words, boxes=boxes, padding='max_length',
                      truncation=True, return_tensors='pt')
    batch_size, _, image_size, _ = batch.pixel_values.size()

    # Generate a mask indicating which tokens to mask for MLM
    seq_length = processor.tokenizer.model_max_length
    mlm_mask = make_mlm_mask(seq_length, batch_size)
    special_tokens_mask = torch.tensor(
        [encoding.special_tokens_mask for encoding in batch.encodings])
    mlm_mask = (mlm_mask & ~special_tokens_mask).bool()

    # To create the MLM task labels clone the input ids and set non-masked ids to the ignore_label
    text_labels = batch.input_ids.clone()
    text_labels.masked_fill_(~mlm_mask, IGNORE_LABEL)
    batch['text_labels'] = text_labels

    # Set the tokens to be masked for MLM to the <MASK> token
    batch.input_ids.masked_fill_(mlm_mask, processor.tokenizer.mask_token_id)

    # TODO where to put this?
    image_mask_generator = MaskingGenerator(16, int(16 * 16 * .4))

    # Generate a mask indicating which tokens to mask for MIM
    mim_mask = torch.tensor(
        np.array([image_mask_generator() for _ in range(batch_size)])).bool()

    # The mask is 16x16 (14x14 in the paper) but the image is 224x224 pixels so up-sample before applying the mask
    upsampled_mim_mask = torch.kron(mim_mask, torch.ones((14, 14))).unsqueeze(-1).bool()
    permuted_image = torch.permute(batch.pixel_values, (0, 2, 3, 1))
    masked_image = torch.permute(permuted_image * ~upsampled_mim_mask, (0, 3, 1, 2))
    batch.pixel_values = masked_image

    # TODO verify that -100 is the correct label for the class token and that it lines up with the class token in
    #  the image token sequence

    wpa_labels = generate_wpa_labels(batch.bbox, upsampled_mim_mask, mlm_mask, IGNORE_LABEL)
    batch['wpa_labels'] = wpa_labels

    image_labels = torch.LongTensor(image_tokens)
    image_labels.masked_fill_(~mim_mask, IGNORE_LABEL)
    image_labels = image_labels.view(batch_size, -1)
    cls_labels = torch.tensor([[IGNORE_LABEL]] * batch_size)
    image_labels = torch.cat([cls_labels, image_labels], dim=1)
    batch['image_labels'] = image_labels

    batch['doc_id'] = torch.LongTensor(doc_ids)

    return batch


def get_datapipe(url, batch_size):
    dps = []
    for filename in IterableWrapper([url]).list_files_by_fsspec():
        dp = FSSpecFileOpener([filename], mode="rb", anon=True)
        dp = dp.load_from_tar(mode="r|")
        dp = dp.read_from_stream()
        dp = dp.webdataset()
        dp = dp.map(sample_page)
        dps.append(dp)

    pipes_to_weights_dict = {dp: 1 / len(dps) for dp in dps}
    data_pipe = SampleMultiplexer(pipes_to_weights_dict=pipes_to_weights_dict, seed=0)
    # data_pipe = data_pipe.batch(batch_size)
    data_pipe = data_pipe.sharding_filter()  # is this the right place for this?
    data_pipe = data_pipe.batch(batch_size)
    data_pipe = data_pipe.map(process_batch)
    return data_pipe


if __name__ == '__main__':
    datapipe = get_datapipe('gs://common-crawl-33-pdf-grouped-english', 10)
    # rs = MultiProcessingReadingService(num_workers=2)
    dl = DataLoader2(datapipe=datapipe)
    for j, batch in enumerate(dl):
        for example in batch:
            print(example['__key__'])
