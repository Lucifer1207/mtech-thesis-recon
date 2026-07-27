e8_codebook.py :- Python script for codebook formation.

export_all_models_v2.py :- The primary purpose of export_all_models_v2.py is to extract, quantize(E8 Lattice Vector Quantiation), and serialize the weights of multiple Transformer models (such as BERT-Base and TinyBERT) into raw binary configuration files (.bin) for cross-architecture or mobile deployment.

usb_debugging.txt :- this is a text file which include steps to allow usb debugging in your android mobile phone.

full_model_benchmark_v2.cpp :- Its primary purpose is to measure and verify how these models behave when executed on a target architecture (such as an Android device via adb).It basically creates ELF file.

E8_Vs-Z8_Distortion.py :- Both E8 and Z8 must use the SAME number of codewords (same bit budget). For a fair test, generate N random 8D vectors, quantize each one using BOTH E8 and Z8, measure reconstruction error for both, compare. RESULTS will be as follows:- 
Bits   E8 SNR    Z8 SNR    E8 Advantage
2      0.603 dB  0.603 dB  -0.000 dB  (tied)
3      1.215 dB  1.146 dB  +0.069 dB  (E8 wins)
4      1.871 dB  1.722 dB  +0.149 dB  (E8 wins clearly)
6      3.002 dB  2.841 dB  +0.161 dB  (E8 wins clearly)
8      4.067 dB  4.179 dB  -0.112 dB  (Z8 wins slightly)

DataSet Details:-
data/anubis/  → 574 JSON files
data/auto/    → 565 JSON files
data/manual/  → 274 JSON files

RAW data:    58,052 flows total, 4,652 leaks, 53,400 non-leaks


ReCon Pipeline Complete Order of Execution for QAD :- Stage1a -> Stage 1b -> Stage 2b -> Stage_e8-qad .

ReCon_pipeline/Stage1a_load_Dataset.py :- The primary purpose of stage1_load_data.py is to transform raw, noisy JSON network logs into a clean, structured, and machine-learning-ready format for training your model.

ReCon_pipeline/Stage1b_demask_data.py :- This script is used to clean the data and remove markers like "RECON_" and output files "train_clean.json", "test_clean.json".

ReCon_pipeline/Stage2b_train_teacher.py :- This file is Fine-Tuning BERT-Base FP32 model. Additionally, it generates "teacher_train_probs.json" , which contains the output probabilities (soft labels) that your Student model will need for distillation.Hence, this model can be used as Teacher Model. The script follows this Train/test/validate split:- 

TRAIN :    41,797 total | PII=3,386  | Non-PII=38,411 
VALIDATE : 4,644 total | PII=356  | Non-PII=4,288 
TEST :     11,611 total | PII=910 | Non-PII=10,701 

ReCon_pipeline/stage_e8_qad.py :- This file is final step of ReCon pipeline. This trains student model(TinyBERT), using E8 Lattice Vector Quantization and using teacher_prob file(that gets generated after Stage2b). The teacher_model used here is:- FineTuned Teacher model(Bert-BASE) on cleaned dataset, that gets saved after Stage2b.

ReCon_pipeline/export_qad_recon_models.py :- this file is used to create .bin for student_model_lattice_e8(that gets created after stage_e8_qad.py run. This student_model_lattice_e8 is trained QAD student(TinyBERT) using teacher as BERT-Base.

ReCon_pipeline/full_model_benchmark_recon.cpp :- Its primary purpose is to measure and verify how these models behave when executed on a target architecture (such as an Android device via adb). It basically creates ELF file.  

FP32_scripts folder :- This folder contains 2 files, one for fp32 tinybert, and one for fp32 bert base.


QAT_Scripts/stage_all_qat.py:- This script performs Quantization-Aware Training (QAT) on BERT-Base and TinyBERT using E8 lattice and Z8 scalar quantization to optimize models for mobile deployment.

QAT_Scripts/export_all_qat_models.py:- While your training script (stage_qat_all.py) learns the weights, it stores them in a format that your mobile device's C++ inference engine cannot understand. This export script converts those trained weights into raw binary files that your C++ benchmark can load directly into memory.

QAT_Scripts/full_model_benchmark_qat.cpp :- The purpose of full_model_benchmark_qat.cpp is to evaluate the actual performance, memory footprint, and numerical accuracy of your quantized transformer models (BERT/TinyBERT) on physical mobile-class hardware. 



QAT_Scripts/stage_all_qat_normalization.py :- The purpose of the stage_qat_all_normalization.py script is to implement theoretically grounded, paper-compliant row-wise dynamic-range calibration for Lattice-based Quantization-Aware Training (QAT).


QAT_Scripts/stage_all_qat_normalization_CLA.py :- This is same file like stage_all_qat_normalization.py, but this uses command line arguement to take epoch count as input from terminal like :- python3 stage_all_qat_normalization_CLA.py --epochs 3.

QAT_Scripts/export_all_qat_models_normalization.py :- This script serves as the bridge between the statistical training domain and the hardware deployment domain. While your training script (stage_qat_all_normalization.py) learns weights using the paper’s row-wise statistical calibration ($\sigma_i$, $\beta_i$), those raw PyTorch tensors are not natively readable by a C++ inference engine.

QAT_Scripts/full_model_benchmark_qat_normalization.cpp :- This script acts as the final "real-world" test to prove that your lattice-based model actually runs efficiently on an Android device.
