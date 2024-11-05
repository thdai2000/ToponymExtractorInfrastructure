The script for preparing datasets for deepsolo model, and grouper model.

## deepsolo data

### pre-training data

- ic13, ic15, mlt2017, totaltext, syntext1&2 (https://github.com/ViTAE-Transformer/DeepSolo/tree/main/DeepSolo)

- synmap (https://knowledge-computing.github.io/mapkurator-doc/#/docs/modules/spot)

### finetuning data

- MapKurator humman annotations (https://knowledge-computing.github.io/mapkurator-doc/#/docs/modules/spot)
- ICDAR 2024 MapText Competition (only "General Data from the David Rumsey Collection") (https://rrc.cvc.uab.es/?ch=28&com=downloads)
- CVAT annotations (20240904 version) (https://github.com/epfl-timemachine/toponymics/tree/main/annotations/final)
- all in one: https://huggingface.co/datasets/thdai2000/ToponymExtractorDatasets/blob/main/finetuning_data.zip (original, unprocessed images)

### pre-processing procedure
1. for ic13, ic15, mlt2017, totaltext, syntext1&2, no pre-processing is needed
2. for synmap and finetuning data
   1. estimate 8 bezier curve control points for each word
   2. encode text with 96 vocab
   3. for finetuning data
      1. make tiles of 2x2 or 3x3
      2. handle marginal samples by
         - proportionally truncate the bounding line and re-estimate bezier control points
         - proportionally truncate the text

All processed datasets (except syntext) for deepsolo can be downloaded from:
https://huggingface.co/datasets/thdai2000/ToponymExtractorDatasets/

## Deepsolo Training tips
1. follow instruction in https://github.com/ViTAE-Transformer/DeepSolo/tree/main/DeepSolo to install the environment
2. go to adet/data/builtin.py and register datasets if needed
3. go to adet/config/defaults.py and change `_C.MODEL.TRANSFORMER.VOC_SIZE` to 96
4. the number of training iterations depends on your compute budget, 20-40 epochs is preferred.

## grouper data

Grouper data is obtained from MapText, MapKurator, and CVAT annotations. Download link: https://huggingface.co/datasets/thdai2000/ToponymExtractorDatasets/blob/main/grouper_data.zip


