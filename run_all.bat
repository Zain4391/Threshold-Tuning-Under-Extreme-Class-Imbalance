@echo off
rem Full batch. Results go to v2_* folders, logs to logs\*.txt. Safe to stop and re-run single lines.
if not exist logs mkdir logs

rem ---------- FRAUD ----------
python exp_ac.py --vary train   --source fraud --seeds 30 --out v2_A_fraud    > logs\A_fraud.txt 2>&1
python exp_ac.py --vary val     --source fraud --seeds 30 --out v2_B_fraud    > logs\B_fraud.txt 2>&1
python exp_ac.py --vary ratio   --source fraud --seeds 30 --out v2_C_fraud    > logs\C_fraud.txt 2>&1
python exp_ac.py --vary ratio_n --source fraud --seeds 30 --out v2_D_fraud    > logs\D_fraud.txt 2>&1
python exp_strat.py --vary ratio_n --source fraud --seeds 30 --out v2_S_D_fraud > logs\S_D_fraud.txt 2>&1
python exp_strat.py --vary train   --source fraud --seeds 30 --out v2_S_A_fraud > logs\S_A_fraud.txt 2>&1
python exp_strat.py --vary ratio   --source fraud --seeds 30 --out v2_S_C_fraud > logs\S_C_fraud.txt 2>&1

rem ---------- COVERTYPE (ratios stop at 1:200) ----------
python exp_ac.py --vary train   --source covtype --seeds 30 --out v2_A_cov > logs\A_cov.txt 2>&1
python exp_ac.py --vary val     --source covtype --seeds 30 --out v2_B_cov > logs\B_cov.txt 2>&1
python exp_ac.py --vary ratio   --source covtype --grid 10 20 50 100 200 --seeds 30 --out v2_C_cov > logs\C_cov.txt 2>&1
python exp_ac.py --vary ratio_n --source covtype --grid 20 50 100 200    --seeds 30 --out v2_D_cov > logs\D_cov.txt 2>&1
python exp_strat.py --vary ratio_n --source covtype --grid 20 50 100 200    --seeds 30 --out v2_S_D_cov > logs\S_D_cov.txt 2>&1
python exp_strat.py --vary train   --source covtype --seeds 30 --out v2_S_A_cov > logs\S_A_cov.txt 2>&1
python exp_strat.py --vary ratio   --source covtype --grid 10 20 50 100 200 --seeds 30 --out v2_S_C_cov > logs\S_C_cov.txt 2>&1

rem ---------- PROTEIN_HOMO (ratios stop at 1:100) ----------
python exp_ac.py --vary train   --source protein_homo --seeds 30 --out v2_A_prot > logs\A_prot.txt 2>&1
python exp_ac.py --vary val     --source protein_homo --seeds 30 --out v2_B_prot > logs\B_prot.txt 2>&1
python exp_ac.py --vary ratio   --source protein_homo --grid 10 20 50 100 --seeds 30 --out v2_C_prot > logs\C_prot.txt 2>&1
python exp_ac.py --vary ratio_n --source protein_homo --grid 20 50 100    --seeds 30 --out v2_D_prot > logs\D_prot.txt 2>&1
python exp_strat.py --vary ratio_n --source protein_homo --grid 20 50 100    --seeds 30 --out v2_S_D_prot > logs\S_D_prot.txt 2>&1
python exp_strat.py --vary train   --source protein_homo --seeds 30 --out v2_S_A_prot > logs\S_A_prot.txt 2>&1
python exp_strat.py --vary ratio   --source protein_homo --grid 10 20 50 100 --seeds 30 --out v2_S_C_prot > logs\S_C_prot.txt 2>&1
echo ALL DONE