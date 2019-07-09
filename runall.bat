echo "Start of Run All Script"
:: Exp on features GRU

:: SW (30 or 40)
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 10 -bz 64 -hs 256 -lt 2 -exp 1 > ./console_outputs/exp1.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 20 -bz 64 -hs 256 -lt 2 -exp 2 > ./console_outputs/exp2.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 30 -bz 64 -hs 256 -lt 2 -exp 3 > ./console_outputs/exp3.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 64 -hs 256 -lt 2 -exp 4 > ./console_outputs/exp4.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 50 -bz 64 -hs 256 -lt 2 -exp 5 > ./console_outputs/exp5.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 60 -bz 64 -hs 256 -lt 2 -exp 6 > ./console_outputs/exp6.txt

:: Batch Size 32
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 16 -hs 256 -lt 2 -exp 7 > ./console_outputs/exp7.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 256 -lt 2 -exp 8 > ./console_outputs/exp8.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 128 -hs 256 -lt 2 -exp 9 > ./console_outputs/exp9.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 256 -hs 256 -lt 2 -exp 10 > ./console_outputs/exp10.txt

:: Hidden Size 128
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 64 -lt 2 -exp 11 > ./console_outputs/exp11.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -lt 2 -exp 12 > ./console_outputs/exp12.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 512 -lt 2 -exp 13 > ./console_outputs/exp13.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 1024 -lt 2 -exp 14 > ./console_outputs/exp14.txt

:: Num Layers 1 layer or 3 layers equivalent
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 2 -lt 2 -exp 15 > ./console_outputs/exp15.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 16 > ./console_outputs/exp16.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 4 -lt 2 -exp 17 > ./console_outputs/exp17.txt

:: Testing with other labels 3 layers is better
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 1 -lt 3 -exp 18 > ./console_outputs/exp18.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 1 -lt 4 -exp 19 > ./console_outputs/exp19.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 3 -lt 3 -exp 20 > ./console_outputs/exp20.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 128 -nl 3 -lt 4 -exp 21 > ./console_outputs/exp21.txt

:: Testing with more contiguous datasets
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 2 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 47 > ./console_outputs/exp47.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 3 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 48 > ./console_outputs/exp48.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 4 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 49 > ./console_outputs/exp49.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 5 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 50 > ./console_outputs/exp50.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 6 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 51 > ./console_outputs/exp51.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 7 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 52 > ./console_outputs/exp52.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 8 -sw 40 -bz 32 -hs 128 -nl 3 -lt 2 -exp 53 > ./console_outputs/exp53.txt

:: Exp on base GRU

:: SW 80 is the only not 100% unbalanced
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 10 -bz 64 -hs 256 -lt 2 -exp 22 > ./console_outputs/exp22.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 20 -bz 64 -hs 256 -lt 2 -exp 23 > ./console_outputs/exp23.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 30 -bz 64 -hs 256 -lt 2 -exp 24 > ./console_outputs/exp24.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 64 -hs 256 -lt 2 -exp 25 > ./console_outputs/exp25.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 50 -bz 64 -hs 256 -lt 2 -exp 26 > ./console_outputs/exp26.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 70 -bz 64 -hs 256 -lt 2 -exp 27 > ./console_outputs/exp27.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 80 -bz 64 -hs 256 -lt 2 -exp 28 > ./console_outputs/exp28.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 90 -bz 64 -hs 256 -lt 2 -exp 29 > ./console_outputs/exp29.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 100 -bz 64 -hs 256 -lt 2 -exp 30 > ./console_outputs/exp30.txt

:: Batch Size, all bad, bz 32
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 80 -bz 16 -hs 256 -lt 2 -exp 31 > ./console_outputs/exp31.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 80 -bz 32 -hs 256 -lt 2 -exp 32 > ./console_outputs/exp32.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 80 -bz 128 -hs 256 -lt 2 -exp 33 > ./console_outputs/exp33.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 80 -bz 256 -hs 256 -lt 2 -exp 34 > ./console_outputs/exp34.txt

:: Hidden Size & layers
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 64 -lt 2 -exp 35 > ./console_outputs/exp35.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 128 -lt 2 -exp 36 > ./console_outputs/exp36.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 512 -lt 2 -exp 37 > ./console_outputs/exp37.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 1024 -lt 2 -exp 38 > ./console_outputs/exp38.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 2048 -lt 2 -exp 39 > ./console_outputs/exp39.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 4096 -lt 2 -exp 40 > ./console_outputs/exp40.txt

::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 64 -nl 2 -lt 2 -exp 41 > ./console_outputs/exp41.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 128 -nl 2 -lt 2 -exp 42 > ./console_outputs/exp42.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 512 -nl 2 -lt 2 -exp 43 > ./console_outputs/exp43.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 1024 -nl 2 -lt 2 -exp 44 > ./console_outputs/exp44.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 2048 -nl 2 -lt 2 -exp 45 > ./console_outputs/exp45.txt
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 40 -bz 32 -hs 4096 -nl 2 -lt 2 -exp 46 > ./console_outputs/exp46.txt

:: Exp on features FCL
::python redux.py -m train -mm fcl -ml features_fcl -fa 31 -ad 1 -bz 32 -lt 2 -exp 54  > ./console_outputs/exp54.txt
::python redux.py -m train -mm fcl -ml features_fcl -fa 31 -ad 1 -bz 32 -lt 2 -st oversampling -exp 55  > ./console_outputs/exp55.txt

:: Rerun of some of the "best" models with oversampling
::python redux.py -m train -mm gru -ml base_gru -fa 1 -ad 1 -sw 80 -bz 32 -hs 512 -lt 2 -st oversampling -exp 56 > ./console_outputs/exp56.txt
::python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 256 -lt 3 -st oversampling -exp 57 > ./console_outputs/exp57.txt
python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 256 -lt 2 -st oversampling -exp 58 > ./console_outputs/exp58.txt
python redux.py -m train -mm gru -ml features_gru -fa 31 -ad 1 -sw 40 -bz 32 -hs 256 -lt 4 -st oversampling -exp 59 > ./console_outputs/exp59.txt

echo "End of Run All Script"