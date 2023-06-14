from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
from torchdata.datapipes.iter import SampleMultiplexer, FSSpecFileLister

from datapipe import create_page_example


def demux_classifier(filename):
    klass = int(filename.split('-')[-1].split('.')[0])
    return klass

url = "gs://common-crawl-33-pdf-grouped-english"
dps = FSSpecFileLister(url).demux(7793, demux_classifier, buffer_size=7793)
for i, _ in enumerate(dps):
    dp = dps[i]
    dp = dp.open_files_by_fsspec(mode="rb")
    dp = dp.load_from_tar(mode="r|")
    dp = dp.read_from_stream()
    dp = dp.webdataset()
    dps[i] = dp

pipes_to_weights_dict = {dp: 1 / len(dps) for dp in dps}
datapipe = SampleMultiplexer(pipes_to_weights_dict=pipes_to_weights_dict, seed=0)
datapipe = datapipe.sharding_filter()
datapipe = datapipe.map(create_page_example)
datapipe = datapipe.batch(4)
datapipe = datapipe.process_batch()

rs = MultiProcessingReadingService(num_workers=2)
dl = DataLoader2(datapipe, reading_service=rs)
for epoch in range(10):
    for d in dl:
        print(d.data['doc_id'].tolist())
dl.shutdown()