import wandb
from torch.optim import Adam
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, \
    DistributedReadingService, SequentialReadingService
from transformers import LayoutLMv3Config

from datapipe import get_datapipe
from modeling import LayoutLMv3ForPretraining

USE_GPU = True
NUM_EPOCHS = 10

config = LayoutLMv3Config(
    # The pretrained LayoutLMv3 has max_position_embeddings=514 but the default for LayoutLMv3Config is 512. Because of
    # the way positional indices are assigned a value of 512 here will lead to index errors.
    max_position_embeddings=514,
    # In the LayoutLMv3 paper the authors used a patch size of 16 pixels and an image size of 224x224 pixes.
    # The implementation of dVAE that I used requires images to have dimensions that are a power of 2 so I used images
    # of size 128x128 pixels. The with 3 layers the dVAE down samples each image by a factor of 2^3=8 resulting in an
    # array of 16x16 image patches. I stuck with the 224x224 pixel image size for pretraining the LayoutLMv3 as
    # described in the paper. Dividing a 224x224 pixel image into 16x16 patches requires each patch to have a size of
    # 14x14 pixels.
    patch_size=14
)
config.codebook_size = 8192

run = wandb.init(
    project='train_layoutlm',
    job_type='train_model',
    config=config.to_dict()
)


model = LayoutLMv3ForPretraining(config)
if USE_GPU:
    model = model.cuda()

url = 'gs://common-crawl-33-pdf-grouped-english'

batch_size = 4

datapipe = get_datapipe(url, batch_size)

mp_rs = MultiProcessingReadingService(num_workers=1)
# dist_rs = DistributedReadingService()
# rs = SequentialReadingService(dist_rs, mp_rs)

dataloader = DataLoader2(datapipe=datapipe, reading_service=mp_rs)

opt = Adam(model.parameters())

global_step = 0
for epoch in range(NUM_EPOCHS):
    for i, encoding in enumerate(dataloader):
        if USE_GPU:
            encoding = encoding.to('cuda')
        print(i)
        output = model(**encoding)
        loss = output.mlm_loss + output.mim_loss + output.wpa_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 10 == 0:
            logs = {
                'epoch': epoch,
                'iter': i,
                'mlm loss': float(output.mlm_loss),
                'mim loss': float(output.mim_loss),
                'wpa loss': float(output.wpa_loss),
                'total loss': float(loss),
            }
            wandb.log(logs)

