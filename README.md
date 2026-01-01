# README – Hướng dẫn chạy

## Chuẩn bị môi trường và cài đặt
- git clone https://github.com/amazon-science/GNP.git
- cd GNP
- conda create -n gnp python=3.9
- conda activate gnp
- pip install -r requirements_fixed.txt

## Chạy mô hình trực tiếp
- python llm.py --model roberta-base --task piqa --batch_size 16 --epochs 4 --lr 2e-5

## Chạy bằng script nhanh
bash scripts/llm.piqa.sh

## Lưu logs kết quả
python llm.py --model roberta-base --task piqa --batch_size 16 --epochs 4 --lr 2e-5 | tee logs/piqa_run.txt
