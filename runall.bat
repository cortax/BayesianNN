echo "Start of Run All Script"
:: Comments
python redux.py -m model -mm train -ml hn_gru_net -hs 128 -e 2000 -es 250 -ws 32 -stride 30 -nl 1 -bz 64 --NO_FEATURE_ENDO --NOT_FORECASTING -exp 1 > ./console_outputs/hn_gru_net/exp1.txt
python redux.py -m model -mm train -ml hn_gru_net -hs 128 -e 2000 -es 250 -ws 32 -stride 30 -nl 1 -bz 64 --NO_FEATURE_ENDO --NOT_FORECASTING --RESTART -rm _epoch62_validloss0.0027_model_.torch -exp 1 > ./console_outputs/hn_gru_net/exp1_1.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 256 -e 2000 -es 250 -ws 32 -stride 30 -nl 2 -bz 128 --NO_FEATURE_ENDO --NOT_FORECASTING -exp 2 > ./console_outputs/hn_gru_net/exp2.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 128 -e 2000 -es 250 -ws 32 -stride 30 -nl 1 -bz 128 -exp 3 > ./console_outputs/hn_gru_net/exp3.txt
python redux.py -m model -mm train -ml hn_gru_net -hs 128 -e 1 -es 250 -ws 32 -stride 30 -nl 1 -bz 128 -exp 3 --RESTART -rm _epoch556_validloss0.0000_model_.torch > ./console_outputs/hn_gru_net/exp3_1.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 128 -e 2000 -es 250 -ws 32 -delta 10 -stride 30 -nl 2 -bz 32 -exp 4 > ./console_outputs/hn_gru_net/exp4_0.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 64 -e 2000 -es 250 -ws 32 -stride 30 -nl 1 -bz 32 --NO_FEATURE_ENDO --NOT_FORECASTING -exp 5 > ./console_outputs/hn_gru_net/exp5_0.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 256 -e 2000 -es 250 -ws 32 -stride 30 -nl 1 -bz 32 --NO_FEATURE_ENDO --NOT_FORECASTING -exp 6 > ./console_outputs/hn_gru_net/exp6_0.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 512 -e 2000 -es 250 -ws 32 -stride 30 -nl 1 -bz 32 --NO_FEATURE_ENDO --NOT_FORECASTING -exp 7 > ./console_outputs/hn_gru_net/exp7_0.txt

python redux.py -m model -mm train -ml hn_gru_net -hs 256 -e 2000 -es 250 -ws 32 -stride 30 -nl 2 -bz 32 --NO_FEATURE_ENDO --NOT_FORECASTING -exp 8 > ./console_outputs/hn_gru_net/exp8_0.txt

echo "End of Run All Script"