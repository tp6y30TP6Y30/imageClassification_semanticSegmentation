wget 'https://www.dropbox.com/s/ckcbtlmfhcmd8ky/hw2_1.ckpt?dl=1' -O ./Problem1/hw2_1.ckpt

python3 ./Problem1/main.py --mode test --test_img_path $1 --pred_path $2