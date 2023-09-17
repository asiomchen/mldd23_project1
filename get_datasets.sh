#!/bin/bash
cd data
mkdir train_data
cd train_data
wget -O big_dataset_standardized.parquet https://www.dropbox.com/scl/fi/toutcm07u31krnkk9qoo8/big_dataset_std.parquet?rlkey=37tchxwu1k6ovvi8xsweyobf5&dl=1
# wget -O big_dataset.parquet https://www.dropbox.com/scl/fi/kbrlo5hqvpezmp0v4nlm9/big_dataset.parquet?rlkey=0f28kmf2vy9gs9k57fagdu4k5&dl=1
# wget -O train_morgan_512bits.parquet https://www.dropbox.com/s/bj37jlql9j1lcrr/train_morgan_512bits.parquet?dl=1
cd -
mkdir activity_data
cd activity_data
# wget -O 5ht1a_klek_100nM.parquet https://www.dropbox.com/scl/fi/surb3lv87yrwcfcu6evn2/5ht1a_klek_100nM.parquet?rlkey=edqpteyt3m9e3cump9pguwvfa&dl=1
# wget -O 5ht7_klek_100nM.parquet https://www.dropbox.com/scl/fi/giuhzk3n191w1opy4bews/5ht7_klek_100nM.parquet?rlkey=9qs1u906u39rqm7u6kco3wldh&dl=1
# wget -O beta2_klek_100nM.parquet https://www.dropbox.com/scl/fi/4hvmt4wstyhtynghta22k/beta2_klek_100nM.parquet?rlkey=q6u6dbzva3iqx1sfs3mc22eda&dl=1
# wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/b7xtwokibqcetlgdomuiy/d2_klek_100nM.parquet?rlkey=3v3uet9c5jov059zowds4h9a7&dl=1
# wget -O h1_klek_100nM.parquet https://www.dropbox.com/scl/fi/ctlumo9t6gf4zc1hrronl/h1_klek_100nM.parquet?rlkey=1e7z8a6alah4pdljbtw1378tl&dl=1
# cd -
# mkdir interactions
# cd interactions
# wget -O 5ht1a_dense.parquet https://www.dropbox.com/s/lcwn20vlc2xq5i4/5ht1a_dense.parquet?dl=1
# wget -O 5ht7_dense.parquet https://www.dropbox.com/s/pg42p1jqy4ghrwi/5ht7_dense.parquet?dl=1
# wget -O beta2_dense.parquet https://www.dropbox.com/s/1zsekovgsy2vfik/beta2_dense.parquet?dl=1
# wget -O d2_dense.parquet https://www.dropbox.com/s/fb4tgn92wgshs05/d2_dense.parquet?dl=1
# wget -O h1_dense.parquet https://www.dropbox.com/s/wt14xi0elqxqa08/h1_dense.parquet?dl=1
echo All datasets downloaded successfully
