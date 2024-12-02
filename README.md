# 1 Architecture
![Architecture](https://github.com/xxx803/KambaAD/blob/main/figures/Architecture.png)
# 2 Result

**2.01** Overall performance comparison of KambaAD and baseline models across four real-world
multivariate datasets: SMD, MSL, SMAP, and PSM. Models are ranked from lowest to highest
performance. Precision (P), Recall (R), and F1-score (F1) are reported in percentages (%). The best
performance in each metric is highlighted in bold, and the second-best is underlined. A dash (-)
indicates that the model’s result is missing for the specific dataset.

![figure01](https://github.com/xxx803/KambaAD/blob/main/figures/figure01.png)

**2.02** Multi-metric performance comparison of KambaAD, DCdetector, and AnomalyTrans-
former on the NIPS dataset. Aff-P and Aff-R denote the precision and recall for the affiliation
metric. R A R (Range AUC ROC) and R A P (Range AUC PR) represent scores based on label
transformation under the ROC and PR curves, respectively. V ROC and V RR correspond to the
volumes under the ROC and PR curve surfaces. All results are reported in percentages (%). The
best performance in each metric is highlighted in bold, and the second-best is underlined.

![figure02](https://github.com/xxx803/KambaAD/blob/main/figures/figure02.png)

**2.03** Performance comparison between KAN (point) and the proposed KambaAD model (KAN
window) across eight real-world multivariate datasets. Precision (P), Recall (R), and F1-score (F1)
are reported in percentages (%). The best results are highlighted in bold.

![figure03](https://github.com/xxx803/KambaAD/blob/main/figures/figure03.png)

**2.04** Performance comparison between the Encoder-only, Reconstruction-only, and KambaAD
across eight real-world multivariate datasets. Precision (P), Recall (R), and F1-score (F1) are re-
ported in percentages (%). The best results are highlighted in bold, and the second-best results are
underlined.

![figure04](https://github.com/xxx803/KambaAD/blob/main/figures/figure04.png)

**2.05** Performance comparison between KambaAD and five ablation study models across eight
real-world multivariate datasets. Only the comparison results of the F1 score are presented. The best
results are highlighted in bold, and the second-best results are underlined.

![figure05](https://github.com/xxx803/KambaAD/blob/main/figures/figure05.png)

**2.06** Parameter sensitivity studies of main hyper-parameters in KambaAD

![figure06](https://github.com/xxx803/KambaAD/blob/main/figures/figure06.png)

**2.07** Performance comparison between channel-independent (CI) and channel-dependent (CD)
reconstruction methods across eight real-world multivariate datasets. Precision (P), Recall (R), and
F1-score (F1) are reported in percentages (%). The best results are highlighted in bold.

![figure07](https://github.com/xxx803/KambaAD/blob/main/figures/figure07.png)

**2.08** The presented figure illustrates the reconstruction of features 4, 12, 16, and 23 in a data
sample from PSM following the extraction of three crucial components in KambaAD, along with the
identification of anomalous points based on their reconstructed values using an optimal threshold.

![figure08](https://github.com/xxx803/KambaAD/blob/main/figures/figure08.png)

**2.09** Dataset Statistics.

![figure09](https://github.com/xxx803/KambaAD/blob/main/figures/figure09.png)

**2.10** The common hyperparameter settings used for training the model across all datasets.

![figure10](https://github.com/xxx803/KambaAD/blob/main/figures/figure10.png)

**2.11**The dataset-specific hyperparameter settings used for training the model on different
datasets.

![figure11](https://github.com/xxx803/KambaAD/blob/main/figures/figure11.png)

**2.12** Performance comparison of three models with different component orders: KAN-
MAMBA-attention, Attention-MAMBA-KAN, and KambaAD (KAN-attention-MAMBA) across
eight real-world multivariate datasets. Precision (P), Recall (R), and F1-score (F1) are reported in
percentages (%). The best results are highlighted in bold, and the second-best results are underlined.

![figure12](https://github.com/xxx803/KambaAD/blob/main/figures/figure12.png)

**2.13** Comparison of anomaly scores from KambaAD, DCdetector, and AnomalyTransformer
on the same data segment. The upper panel shows time series features with anomalies in red, while
the lower panel presents the models’ anomaly scores, also highlighting detected anomalies in red.

![figure13](https://github.com/xxx803/KambaAD/blob/main/figures/figure13.png)

**2.14** Performance comparison between the Encoder-only, Reconstruction-only, and KambaAD
models across eight real-world multivariate datasets, with the model sizes kept approximately equiv-
alent. Precision (P), Recall (R), and F1-score (F1) are reported in percentages (%). The best results
are highlighted in bold, and the second-best results are underlined.

![figure14](https://github.com/xxx803/KambaAD/blob/main/figures/figure14.png)

**2.15** Comprehensive Computational Efficiency Analysis of KambaAD, AnomalyTransformer,
and DCdetector: A Comparative Study across MSL, SMAP, SMD, and PSM Datasets. Metrics
include Training Time (seconds), GPU Expend (MB), Memory Expend (MB), Model Size (MB),
and Parameter Count (millions).

![figure15](https://github.com/xxx803/KambaAD/blob/main/figures/figure15.png)

# 3 development environment
    python==3.12.5
    torch==2.2.1+cu118
    torchaudio==2.2.1+cu118
    torchinfo==1.8.0
    numpy==1.26.4
    matplotlib==3.9.2
    causal-conv1d1.2.0.post1
    mamba-ssm==1.2.0
    pandas2.2.3
    
Other components can be installed directly with pip. At the same time, you can parameter the requirements.txt file in the code, which is directly exported from the open environment.

# 4 Dataset
The dataset is located in the datasource directory and includes eight groups: MSL, SMAP, SMD, PSM, NIPS_TS_CCard, NIPS_TS_Swan, NIPS_TS_GECCO, and NIPS_TS_Syn_Mulvar. Simply extract them into the current directory to access. If you place them in another directory, please modify the processed_path parameter in src/arguments.py accordingly.

# 5 Model training parameter
The training parameters for each mode are in the results directory. Extract them into the original directory. It's important to note that due to GitHub's space limitations, we couldn't upload all the training data. We only included the first set of data for MSL, SMAP, and SMD. The training parameters for PSM, NIPS_TS_CCard, NIPS_TS_Swan, NIPS_TS_GECCO, and NIPS_TS_Syn_Mulvar are complete. If you need to see the training results for MSL, SMAP, and SMD, please train them yourself. During training, modify the files in the get_dataset_info function in src/supports/constants.py by disabling single files and enabling multiple files.
# 6 Deployment procedure
## 6.1 Deploy the environment according to Section 3
## 6.2 unzip the data according to sections 4 and 5 and copy it to the project directory
## 6.3 run run.py directly