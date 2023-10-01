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
wget -O 5ht7_klek_100nM.parquet https://www.dropbox.com/scl/fi/aldgmxtfkj1377st10y0j/5ht7_klek_100nM_std.parquet?rlkey=6xw3ls3bdlppdqifs6tca6j2q&dl=1
wget -O 5ht7_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/i4cw3f6hk2yrbq4g4dvm9/5ht7_ECFP_100nM_std.parquet?rlkey=tug125ls9xnbe9m49e5dlhnb2&dl=1
wget -O beta2_klek_100nM.parquet https://www.dropbox.com/scl/fi/mshoa1qj4vta0v9q4ckg6/beta2_klek_100nM_std.parquet?rlkey=424efiv5eyjrotgfjvqoqj7rl&dl=1
wget -O beta2_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/nsmwkuzzyt0jj36on5st4/beta2_ECFP_100nM_std.parquet?rlkey=7fxxka4divp1ipup0c4ciq7p7&dl=1
wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/k96gdlxle87b3qak2f3g8/d2_klek_100nM_std.parquet?rlkey=etdw303bko452l3nilxtqa1gn&dl=1
wget -O d2_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/qt1znjo79tffrdp0tx4dy/d2_ECFP_100nM_std.parquet?rlkey=dg3ie6zaw9k3lun1cefyk3597&dl=1
wget -O h1_klek_100nM.parquet https://www.dropbox.com/scl/fi/1tr0mobxwd2wn9ebwwpxx/h1_klek_100nM_std.parquet?rlkey=b0jry1piueke17dax9luu09j1&dl=1
wget -O h1_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/80cl5qr2n6chkxw2x4j2j/h1_ECFP_100nM_std.parquet?rlkey=6pvxp3km9okett4rlj3k7dfv0&dl=1
echo All datasets downloaded successfully
