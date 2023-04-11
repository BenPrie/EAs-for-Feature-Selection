# EAs-for-Feature-Selection
We present GeneSift for logistic regression tasks. It can easily be modified for other tasks by changing the fitness function inside the script `featuer_selection.py`, which defines the GeneSift class.

We have demonstrated GeneSift's performance on two datasets (given in `breast_cancer_data.csv` and `dry_bean_data.csv`), giving the results in corresponding Jupyter notebook files.

Be mindful that GeneSift optimises through random evolution, so not all runs are necessarily equal. For the case of breast cancer diagnosis, GeneSift's performance tends to fluctate between 0.95 and 0.98, whereas dry bean classification is very stable.
