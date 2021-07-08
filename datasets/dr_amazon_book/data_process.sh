#! /bin/bash
set -e
echo "begin download data"
mkdir raw_data
cd raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_10.json.gz
gzip -d reviews_Books_10.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gzip -d meta_Books.json.gz
echo "download data successfully"

cd ..
python convert_pd.py
python remap_id.py
