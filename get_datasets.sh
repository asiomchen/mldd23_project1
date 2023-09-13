#!/bin/bash
cd data
mkdir train_data
cd train_data
wget -O big_dataset.parquet https://www.dropbox.com/scl/fi/kbrlo5hqvpezmp0v4nlm9/big_dataset.parquet?rlkey=0f28kmf2vy9gs9k57fagdu4k5&dl=1
# wget -O train_morgan_512bits.parquet https://www.dropbox.com/s/bj37jlql9j1lcrr/train_morgan_512bits.parquet?dl=1
# cd -
# mkdir interactions
# cd interactions
# wget -O 5ht1a_dense.parquet https://www.dropbox.com/s/lcwn20vlc2xq5i4/5ht1a_dense.parquet?dl=1
# wget -O 5ht7_dense.parquet https://www.dropbox.com/s/pg42p1jqy4ghrwi/5ht7_dense.parquet?dl=1
# wget -O beta2_dense.parquet https://www.dropbox.com/s/1zsekovgsy2vfik/beta2_dense.parquet?dl=1
# wget -O d2_dense.parquet https://www.dropbox.com/s/fb4tgn92wgshs05/d2_dense.parquet?dl=1
# wget -O h1_dense.parquet https://www.dropbox.com/s/wt14xi0elqxqa08/h1_dense.parquet?dl=1
echo All datasets downloaded successfully
