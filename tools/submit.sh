#!/bin/bash
 
#------------------------- parametres --------------------------------------
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30      #nb de coeurs
#SBATCH --partition=normal    #partition
#SBATCH --mem=64000            #RAM en Mo
 
#-------------------------  ligne de commande  -----------------------------------------
#ATTENTION : le nombre de threads doit correspondre à la valeur --cpus-per-task fixee ci dessus

eval "$(conda shell.bash hook)"
source activate LP
# python tools/qap_cli.py run --module rlt1_volume
python tools/qap_cli.py run --module rlt1_scip