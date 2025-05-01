# RT-DETRv2

## Setup

```shell

pip install -r requirements.txt
```

## Inference

change the annotations_comb_re_val.json path in the yml file below

1. validation set
```shell

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bread_val.yml -r output/rtdetrv2_r18vd_120e_bread/best_comb_split_1e_4.pth --test-only
```

change the annotations_comb_re_test.json path in the yml file below

2. test set
```shell

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bread_test.yml -r output/rtdetrv2_r18vd_120e_bread/best_comb_split_1e_4.pth --test-only
```

3. kisan metric

add the lines (60~66) in src/zoo/rtdetr/rtdetr_postprocessor.py

the last 2 lines in the inference result (Average TP rate) is the kisan metric with the confidence threshold
