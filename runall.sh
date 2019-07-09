echo "Start of Run All Script"
#python redux.py -m train -mm gru -exp 1 -ad 1 -sw 20 | tee -a ./console_outputs/exp1.txt
#python redux.py -m train -mm gru -exp 2 -ad 1 -sw 30 | tee -a ./console_outputs/exp2.txt
#python redux.py -m train -mm gru -exp 3 -ad 1 -sw 40 | tee -a ./console_outputs/exp3.txt
#python redux.py -m train -mm gru -exp 4 -ad 1 -sw 50 | tee -a ./console_outputs/exp4.txt
#python redux.py -m train -mm gru -exp 5 -ad 1 -sw 60 | tee -a ./console_outputs/exp5.txt
#python redux.py -m train -mm gru -exp 6 -ad 1 -hs 32 | tee -a ./console_outputs/exp6.txt
#python redux.py -m train -mm gru -exp 7 -ad 1 -hs 64 | tee -a ./console_outputs/exp7.txt
#python redux.py -m train -mm gru -exp 8 -ad 1 -hs 256 | tee -a ./console_outputs/exp8.txt

#hidden size 512 is around the best
#python redux.py -m train -mm gru -exp 9 -ad 1 -hs 512 -nw 2 | tee -a ./console_outputs/exp9.txt

#batch size 64 is around the best pick
#python redux.py -m train -mm gru -exp 10 -ad 1 -hs 512 --BATCH_SIZE 16 | tee -a ./console_outputs/exp10.txt
#python redux.py -m train -mm gru -exp 11 -ad 1 -hs 512 --BATCH_SIZE 32 | tee -a ./console_outputs/exp11.txt
#python redux.py -m train -mm gru -exp 12 -ad 1 -hs 512 --BATCH_SIZE 128 | tee -a ./console_outputs/exp12.txt
#python redux.py -m train -mm gru -exp 13 -ad 1 -hs 512 --BATCH_SIZE 256 | tee -a ./console_outputs/exp13.txt
#8 worker works, but sometimes problems. sticking to 4

#failed
#python redux.py -m train -mo gru -exp 14 -ad 1 -hs 1024 -nw 2 | tee -a ./console_outputs/exp14.txt

#changing early stop to 40
#python redux.py -m train -mm gru -exp 15 -ad 1 -lt 2 | tee -a ./console_outputs/exp15.txt
#python redux.py -m train -mm gru -exp 16 -ad 1 -lt 3 | tee -a ./console_outputs/exp16.txt
#python redux.py -m train -mm gru -exp 17 -ad 1 -lt 4 | tee -a ./console_outputs/exp17.txt

#python redux.py -m train -mm gru -exp 18 -ad 2 | tee -a ./console_outputs/exp18.txt
#python redux.py -m train -mm gru -exp 19 -ad 3 | tee -a ./console_outputs/exp19.txt
#python redux.py -m train -mm gru -exp 20 -ad 4 | tee -a ./console_outputs/exp20.txt
#python redux.py -m train -mm gru -exp 21 -ad 5 | tee -a ./console_outputs/exp21.txt
python redux.py -m train -mm gru -ml features_gru -bz 2 -sw 4 -exp 0 -ad 1 > ./console_outputs/exp0.txt
