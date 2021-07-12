wget https://paddlerec.bj.bcebos.com/datasets/amazonBookDr/drbook_train_v1.txt
wget https://paddlerec.bj.bcebos.com/datasets/amazonBookDr/drbook_test_v1.txt

mkdir drtrain
mkdir drtest
cp drbook_train_v1.txt drtrain/
cp drbook_test_v1.txt drtest/
rm -f drbook_test_v1.txt drbook_train_v1.txt
