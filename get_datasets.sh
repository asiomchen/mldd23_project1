#!/bin/bash
cd data
mkdir train_data
cd train_data
wget -O big_dataset_std.parquet https://www.dropbox.com/scl/fi/mwohd2i25m7xcygaay3hr/big_dataset_std.parquet?rlkey=zaq8er9ez0ripvc4jm0u0vr1y&dl=1
wget -O big_dataset_ECFP.paruet https://www.dropbox.com/scl/fi/l24c0qd2tfbjn2p9sukdy/big_dataset_ECFP.parquet?rlkey=vjxtauvbtxasiaxqvjhgzafzf&dl=1
# wget -O train_morgan_512bits.parquet https://www.dropbox.com/s/bj37jlql9j1lcrr/train_morgan_512bits.parquet?dl=1
cd -
mkdir activity_data
cd activity_data
# wget -O 5ht1a_klek_100nM.parquet https://www.dropbox.com/scl/fi/surb3lv87yrwcfcu6evn2/5ht1a_klek_100nM.parquet?rlkey=edqpteyt3m9e3cump9pguwvfa&dl=1
# wget -O 5ht7_klek_100nM.parquet https://www.dropbox.com/scl/fi/giuhzk3n191w1opy4bews/5ht7_klek_100nM.parquet?rlkey=9qs1u906u39rqm7u6kco3wldh&dl=1
# wget -O beta2_klek_100nM.parquet https://www.dropbox.com/scl/fi/4hvmt4wstyhtynghta22k/beta2_klek_100nM.parquet?rlkey=q6u6dbzva3iqx1sfs3mc22eda&dl=1
# wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/b7xtwokibqcetlgdomuiy/d2_klek_100nM.parquet?rlkey=3v3uet9c5jov059zowds4h9a7&dl=1
# wget -O h1_klek_100nM.parquet https://www.dropbox.com/scl/fi/ctlumo9t6gf4zc1hrronl/h1_klek_100nM.parquet?rlkey=1e7z8a6alah4pdljbtw1378tl&dl=1
echo All datasets downloaded successfully
