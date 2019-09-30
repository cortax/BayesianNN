echo "Start of Run All Script"
:: Comments
python redux.py -m model -mm train -ml hn_gru_net -ws 32 -stride 30 -nl 2 -bz 256 --NO_FEATURE_ENDO -exp 0 > ./console_outputs/hn_gru_net/exp0.txt
python redux.py -m model -mm train -ml hn_gru_net -ws 32 -stride 30 -nl 2 -bz 256 -e 1 --NO_FEATURE_ENDO -exp 0 --RESTART  -rm _epoch40_validloss0.0026_.torch

python redux.py -m model -mm train -ml hn_gru_net -hs 128 -ws 32 -stride 30 -nl 2 -bz 128 -e 2000 -es 250 --NO_FEATURE_ENDO -exp 0 > ./console_outputs/hn_gru_net/exp0.txt
python redux.py -m model -mm train -ml hn_gru_net -hs 128 -ws 32 -stride 30 -nl 2 -bz 128 -e 1 -es 250 --NO_FEATURE_ENDO -exp 0 --RESTART -rm _epoch1_validloss0.0053_model_.torch



echo "End of Run All Script"