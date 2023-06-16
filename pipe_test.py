from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2

from datapipe import get_datapipe

datapipe = get_datapipe('gs://common-crawl-33-pdf-grouped-english', 4)
rs = MultiProcessingReadingService(num_workers=2)
dl = DataLoader2(datapipe, reading_service=rs)
for epoch in range(10):
    for d in dl:
        print(d.data['doc_id'].tolist())
dl.shutdown()