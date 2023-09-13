#!/bin/bash
cd data
mkdir train_data
cd train_data
wget -O big_dataset.parquet https://www.dropbox.com/scl/fi/kbrlo5hqvpezmp0v4nlm9/big_dataset.parquet?rlkey=0f28kmf2vy9gs9k57fagdu4k5&dl=1
wget -O train_morgan_512bits.parquet https://www.dropbox.com/s/bj37jlql9j1lcrr/train_morgan_512bits.parquet?dl=1
cd -
mkdir activity_data
cd activity_data
wget -O 5ht1a_klek_100nM.parquet https://www.dropbox.com/scl/fi/lsxxodjc596x8hvd0qoe1/5ht1a_klek_100nM.parquet?rlkey=2yay5wgpdupanxe9xatngxaqy&dl=1
wget -O 5ht7_klek_100nM.parquet https://www.dropbox.com/scl/fi/rsqjoza241xls5w3qae26/5ht7_klek_100nM.parquet?rlkey=rses4n02n8h9f7kgg5k0o80vr&dl=1
wget -O beta2_klek_100nM.parquet https://www.dropbox.com/scl/fi/hy1dcderq2xj1zhwdqyz3/beta2_klek_100nM.parquet?rlkey=394al0sfeyrhstfeb5d879tou&dl=1
wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/srqqwjyfmntiiuxfzf3cf/d2_klek_100nM.parquet?rlkey=a8vi1brbilfl2oelxcac0ac5f&dl=1
wget -O h1_klek_100nM.parquet https://www.dropbox.com/scl/fi/qxp3q5gnins6tgzjfnfol/h1_klek_100nM.parquet?rlkey=q4dsjg79429p96x7d7kn6k38p&dl=1
# cd -
# mkdir interactions
# cd interactions
# wget -O 5ht1a_dense.parquet https://www.dropbox.com/s/lcwn20vlc2xq5i4/5ht1a_dense.parquet?dl=1
# wget -O 5ht7_dense.parquet https://www.dropbox.com/s/pg42p1jqy4ghrwi/5ht7_dense.parquet?dl=1
# wget -O beta2_dense.parquet https://www.dropbox.com/s/1zsekovgsy2vfik/beta2_dense.parquet?dl=1
# wget -O d2_dense.parquet https://www.dropbox.com/s/fb4tgn92wgshs05/d2_dense.parquet?dl=1
# wget -O h1_dense.parquet https://www.dropbox.com/s/wt14xi0elqxqa08/h1_dense.parquet?dl=1
echo All datasets downloaded successfully
