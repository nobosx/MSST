# Improved Differential-Neural Attacks on Speck Using Modular Subtraction Splitting Technique

This repository contains the code written in python of improved differential-neural attacks on Speck using **modular subtraction splitting technique** in the paper *Time Complexity Reduction Framework for Attacks on ARX Ciphers*.

There are codes of the following experiments in this repository:

* Attacks on Speck64 (in the folder `./Speck64`):

  1. `test_key_rank.py` contains the code of the (7+1)-round auxiliary practical attack in Section 8.3 of the original paper. Run this script to verify the (7+1)-round attack on Speck64 for valid structures and invalid structures respectively.
  2. `attack_guess_one_round_table.py` contains the code of the (1+1+7+1)-round practical attack on Speck64. A 1-round differential with a probability of 2^-2 is prepended before the 7-round distinguisher.
  3. `attack_guess_two_rounds_table.py` contains the code of the (6+2)-round auxiliary practical attack as well as the (1+3+6+2)-round theoretical attack on Speck64 in Section 8.3 of the original paper. In the (1+3+6+2)-round attack, a 3-round differential with a probability of 2^-12 is placed before the 6-round distinguisher.

* Attacks on Speck96 (in the folder `./Speck96`):

  `attack_guess_two_rounds_table.py` contains the code of the (7+2)-round practical attack on Speck96 in which a 7-round distinguisher is used and the 2-round key guessing procedure is applied. Run this script to verify this attack for valid structures and invalid structures respectively.

* Attacks on Speck128 (in the folder `./Speck128`):

  `attack_guess_two_rounds_table.py` contains the code of the (9+2)-round practical attack on Speck128 in which a 9-round distinguisher is used and the 2-round key guessing procedure is applied. Run this script to verify this attack for valid structures and invalid structures respectively.

If you want to see the output of these attacks, please refer to the subfolder `attack_output` in the respective attack folder. Also, the subfolder `attack_res` holds the log files in the format of numpy arrays of each attacks, such as the running time and the number of surviving key guesses of each attack.

We provide two versions of attack codes, using neural network and its corresponding lookup table as the distinguisher, with filenames suffixed with `_net` and `_table` respectively. We recommend you to run the codes with the `_table` suffix in order to get the attack results faster.

## Results

The following results were obtained on a computer equipped with an Intel Xeon Gold 5218R CPU and a NVIDIA GeForce RTX 3090 GPU. The GPU was used only in the (7+1)-round attack and (1+1+7+1)-round attack on Speck64.

Abbreviations in the following tables:

* #Trails: Number of Trails.
* M: Number of Ciphertext Pairs in Each Structure.
* #NBs: Number of Used Neutral Bits.
* S_th: Threshold for Filtering Key Guess Scores.
* KSR: Key-Surviving Rate.
* SR: Success Rate.
* ART: Average Running Time of Each Trail.
* ASC: Average Score of Correct Key.
* MSC: Minimum Score of Correct Key.
* ASW: Average Score of Wrong Keys.
* ANUS: Average Number of Used Structures.

### Speck64

Results of the (7+1)-round attack for valid structures:

| M | S_th | #Trials | KSR  |  SR  |  ART   |  ASC   |  MSC  |   ASW   |
| :--: | :--: | :-----: | :--: | :--: | :----: | :----: | :---: | :-----: |
| $2^{10}$ | 10 |  1000   | 100% | 100% | 7.68 s | 195.56 | 53.84 | -176.12 |

Results of the (7+1)-round attack for invalid structures:

| M | S_th | #Trials | KSR  | ART |   ASC   |   ASW   | Maximum Score |
| :--: | :--: | :-----: | :--: | :----: | :-----: | :-----: | :-----------: |
| $2^{10}$ | 10 |  1000   |  0%  | 7.77 s | -183.41 | -182.80 |    -91.99     |

Results of the (1+1+7+1)-round attack:

| #NBs | S_th | #Trials |  KSR  |  SR   |  ART  | ANUS |
| :--: | :--: | :-----: | :---: | :---: | :---: | :--: |
| 10 | 10 |  1000   | 82.2% | 82.1% | 32.52 | 4.25 |

Results of the (6+2)-round attack for valid structures:

| M | S_th | #Trials | KSR  |  SR   |   ART   |  ASC   |  MSC  |   ASW   |
| :--: | :--: | :-----: | :--: | :---: | :-----: | :----: | :---: | :-----: |
| $2^{10}$ | 10 |  1000   | 100% | 88.2% | 19.35 s | 259.31 | 37.35 | -312.07 |

Results of the (6+2)-round attack for invalid structures:

| M | S_th | #Trials | KSR  |   ART   |   ASC   |   ASW   | Maximum Score |
| :--: | :--: | :-----: | :--: | :-----: | :-----: | :-----: | :-----------: |
| $2^{10}$ | 10 |  1000   |  0%  | 19.34 s | -314.43 | -314.21 |    -105.54    |

Results of the (1+3+6+2)-round attack:

| #NBs | S_th | #Trials | KSR  |  SR  |  ART   |  ANUS   |
| :--: | :--: | :-----: | :--: | :--: | :----: | :-----: |
|  10  |  10  |   57    |  36  |  36  | 4.15 h | 2577.96 |

### Speck96

Results of the (7+2)-round attack for valid structures:

|    M     | S_th | #Trials |  KSR  |  SR   |   ART   |  ASC   |  MSC   |   ASW   |
| :------: | :--: | :-----: | :---: | :---: | :-----: | :----: | :----: | :-----: |
| $2^{11}$ |  50  |  1000   | 93.7% | 80.9% | 56.57 s | 150.39 | -44.80 | -120.68 |

Results of the (7+2)-round attack for invalid structures:

|    M     | S_th | #Trials | KSR  |   ART   |   ASC   |   ASW   | Maximum Score |
| :------: | :--: | :-----: | :--: | :-----: | :-----: | :-----: | :-----------: |
| $2^{11}$ |  50  |  1000   |  0%  | 57.23 s | -121.08 | -121.75 |     22.03     |

### Speck128

Results of the (9+2)-round attack for valid structures:

|    M     | S_th | #Trials | KSR  |  SR  |   ART    |  ASC  |  MSC   |  ASW   |
| :------: | :--: | :-----: | :--: | :--: | :------: | :---: | :----: | :----: |
| $2^{12}$ |  5   |   100   | 90%  | 78%  | 125.75 s | 46.84 | -21.17 | -97.49 |

Results of the (9+2)-round attack for invalid structures:

|    M     | S_th | #Trials | KSR  |   ART    |  ASC   |  ASW   | Maximum Score |
| :------: | :--: | :-----: | :--: | :------: | :----: | :----: | :-----------: |
| $2^{12}$ |  5   |   100   |  0%  | 125.89 s | -97.52 | -97.91 |     3.79      |
