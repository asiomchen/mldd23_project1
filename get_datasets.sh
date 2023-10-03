#!/bin/bash
cd data
mkdir train_data

cd train_data
wget -O big_dataset_std.parquet wget -O big_dataset_std.parquet https://www.dropbox.com/scl/fi/pdwlyza3l7i56fejl34vq/big_dataset_std.parquet?rlkey=g13z4witdik65x9d4lfzlclpq&dl=1
wget -O big_dataset_ECFP.paruet https://www.dropbox.com/scl/fi/l24c0qd2tfbjn2p9sukdy/big_dataset_ECFP.parquet?rlkey=vjxtauvbtxasiaxqvjhgzafzf&dl=1
wget -O train_morgan_512bits.parquet https://www.dropbox.com/s/bj37jlql9j1lcrr/train_morgan_512bits.parquet?dl=1

cd -
mkdir activity_data
cd activity_data
wget -O 5ht1a_klek_100nM.parquet https://www.dropbox.com/scl/fi/r5evfx8jdifbrn28bz90t/5ht1a_klek_100nM_std.parquet?rlkey=vdwm317zp5f7qxvqlp2j2mxsx&dl=1
wget -O 5ht1a_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/rdf2y3hk20z32io60kgeh/5ht1a_ECFP_100nM_std.parquet?rlkey=but36ys5pk5dq2ipy4xi5yf15&dl=1
wget -O cb1_klek_100nM.parquet https://www.dropbox.com/scl/fi/3wdmdqci7h1w4h6dumcae/cb1_klek_100nM_std.parquet?rlkey=ppi3pett6xos5xztwsrv0maxg&dl=1
wget -O cb1_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/pwzv4wt3tz79u27fq1f9q/cb1_ECFP_100nM_std.parquet?rlkey=81y3fyj8oe88wnmsxvmwj9ybm&dl=1
wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/k96gdlxle87b3qak2f3g8/d2_klek_100nM_std.parquet?rlkey=etdw303bko452l3nilxtqa1gn&dl=1
wget -O d2_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/qt1znjo79tffrdp0tx4dy/d2_ECFP_100nM_std.parquet?rlkey=dg3ie6zaw9k3lun1cefyk3597&dl=1
echo All datasets downloaded successfully
