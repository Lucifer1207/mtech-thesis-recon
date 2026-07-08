e8_codebook.py :- Python script for codebook formation.

export_all_models_v2.py :- The primary purpose of export_all_models_v2.py is to extract, quantize(E8 Lattice Vector Quantiation), and serialize the weights of multiple Transformer models (such as BERT-Base and TinyBERT) into raw binary configuration files (.bin) for cross-architecture or mobile deployment.

usb_debugging.txt :- this is a text file which include steps to allow usb debugging in your android mobile phone.

full_model_benchmark_v2.cpp :- Its primary purpose is to measure and verify how these models behave when executed on a target architecture (such as an Android device via adb).

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
