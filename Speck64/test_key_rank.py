import numpy as np
import speck as sp
import math
from util import extract_sensitive_bits, cal_hw
from time import time
from os import urandom, path, makedirs
from keras.models import load_model

def expand_key_guess_mask(key_guess_bits):
    kg = np.zeros((1,), dtype=np.uint32)
    for kg_bit in key_guess_bits:
        mask = 1 << kg_bit
        kg = np.concatenate((kg, kg ^ mask))
    assert kg[0] == 0
    return kg

def generate_user_key(nr, master_key_bit_length=96):
    WORD_SIZE = sp.WORD_SIZE()
    assert master_key_bit_length % WORD_SIZE == 0
    m = master_key_bit_length // WORD_SIZE
    assert m == 3 or m == 4
    key = np.frombuffer(urandom(m * 4), dtype=np.uint32).reshape(m, 1)
    ks = sp.expand_key(key, nr)
    return ks

def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r

# Try to determine the borrow bit b[kg_low] at bit position kg_low using m consecutive bits [kg_low-1~kg_low-m]
# Return: b[kg_low], is_valid
# is_valid: whether the borrow bit at bit position kg_low of a ciphertext can be recovered
def determine_borrow_bit_from_low(z, y, kg_low, m):
    mask_val = (1 << m) - 1
    a = (z >> (kg_low - m)) & mask_val
    b = (y >> (kg_low - m)) & mask_val
    return a < b, a != b


def test_key_rank(n, net_path, selected_bits, dis_nr, dis_diff, structure_size, kg_low, m, key_guess_bits, right_structure=True, c=None, attack_saved_folder='./'):
    net = load_model(net_path)
    true_key_rank = []
    true_key_score = []
    true_key_valid_pr = []
    wrong_key_score = np.zeros((n, 2**(m+len(key_guess_bits))-1))
    wrong_key_valid_pr = []
    MAX_SUCCEED_KG_DIFF_HW = 2
    sv_whether_succeed = np.zeros(n, dtype=np.uint8)
    sv_whether_surviving_kg = np.zeros(n, dtype=np.uint8)
    sv_attack_time_consumption = np.zeros(n)
    sv_best_kg_diff_hw = []
    sv_best_kg_diff_show_time = np.zeros(1+m+len(key_guess_bits), dtype=np.uint32)
    sv_num_surviving_kgs = np.zeros(n, dtype=np.uint32)
    enc_nr = 1 + dis_nr + 1
    direct_key_guess_space = 2**len(key_guess_bits)
    direct_kg_diff = expand_key_guess_mask(key_guess_bits)
    for attack_index in range(n):
        print("Attack index:", attack_index)
        tk = generate_user_key(enc_nr)
        true_sk = int(tk[-1][0])
        p0l = np.frombuffer(urandom(4 * structure_size), dtype=np.uint32)
        p0r = np.frombuffer(urandom(4 * structure_size), dtype=np.uint32)
        if right_structure:
            p1l = p0l ^ dis_diff[0]; p1r = p0r ^ dis_diff[1]
        else:
            p1l = np.frombuffer(urandom(4 * structure_size), dtype=np.uint32)
            p1r = np.frombuffer(urandom(4 * structure_size), dtype=np.uint32)
        c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, tk)
        c0r = c0l ^ c0r; c1r = c1l ^ c1r
        y0 = sp.ror(c0r, sp.BETA()); y1 = sp.ror(c1r, sp.BETA())
        z0 = c0l ^ true_sk; z1 = c1l ^ true_sk
        # Stage1
        best_kg_diff = 0
        best_kg_diff_score = -100
        num_greater_than_tk = 0
        sk_num = 0
        start_time = time()
        for i in range(2**m):
            additional_kg_diff = i << (kg_low - m)
            tmp_z0 = z0 ^ additional_kg_diff; tmp_z1 = z1 ^ additional_kg_diff
            borrow_bit0, is_valid0 = determine_borrow_bit_from_low(tmp_z0, y0, kg_low, m)
            borrow_bit1, is_valid1 = determine_borrow_bit_from_low(tmp_z1, y1, kg_low, m)
            valid_pair_pos = is_valid0 & is_valid1
            valid_num = np.sum(valid_pair_pos)
            valid_pr = valid_num / structure_size
            if i == 0:
                true_key_valid_pr.append(valid_pr)
            else:
                wrong_key_valid_pr.append(valid_pr)
            tmp_z0 = tmp_z0[valid_pair_pos]; tmp_z1 = tmp_z1[valid_pair_pos]
            tmp_y0 = y0[valid_pair_pos]; tmp_y1 = y1[valid_pair_pos]
            # Stage2
            repeated_direct_kg_diff = np.repeat(direct_kg_diff, valid_num)
            tmp_z0 = np.tile(tmp_z0, direct_key_guess_space) ^ repeated_direct_kg_diff; tmp_z1 = np.tile(tmp_z1, direct_key_guess_space) ^ repeated_direct_kg_diff
            tmp_y0 = np.tile(tmp_y0, direct_key_guess_space); tmp_y1 = np.tile(tmp_y1, direct_key_guess_space)
            x0 = tmp_z0 - tmp_y0; x1 = tmp_z1 - tmp_y1
            x0 = sp.rol(x0, sp.ALPHA()); x1 = sp.rol(x1, sp.ALPHA())
            X = sp.convert_to_binary([x0, tmp_y0, x1, tmp_y1])
            X = extract_sensitive_bits(X, selected_bits)
            Z = net.predict(X, batch_size=10000, verbose=0).reshape(direct_key_guess_space, valid_num); Z = np.log2(Z / (1 - Z))
            Z = np.sum(Z, axis=1) / valid_pr
            if i == 0:
                true_key_score.append(Z[0])
                num_greater_than_tk += np.sum(Z[1:] > true_key_score[-1])
                wrong_key_score[attack_index, :direct_key_guess_space-1] = Z[1:]
            else:
                num_greater_than_tk += np.sum(Z > true_key_score[-1])
                wrong_key_score[attack_index, i*direct_key_guess_space-1:(i+1)*direct_key_guess_space-1] = Z
            sk_num += np.sum(Z > c)
            best_direct_kg_diff_pos = np.argmax(Z)
            if Z[best_direct_kg_diff_pos] > best_kg_diff_score:
                best_kg_diff_score = Z[best_direct_kg_diff_pos]
                best_kg_diff = direct_kg_diff[best_direct_kg_diff_pos] ^ additional_kg_diff
        end_time = time()
        print("Running time: {} s".format(end_time - start_time))
        sv_attack_time_consumption[attack_index] = end_time - start_time
        sv_num_surviving_kgs[attack_index] = sk_num
        true_key_rank.append(num_greater_than_tk)
        if sk_num > 0:
            print("Surviving key guess num:", sk_num)
            sv_whether_surviving_kg[attack_index] = 1
            hw = cal_hw(best_kg_diff, 32)
            sv_best_kg_diff_hw.append(hw)
            sv_best_kg_diff_show_time[hw] += 1
            print("Best key guess score:", best_kg_diff_score)
            print("Diff between tk and best kg:", hex(best_kg_diff))
            if hw <= MAX_SUCCEED_KG_DIFF_HW:
                sv_whether_succeed[attack_index] = 1
        else:
            print("No surviving key guess")
        print('', flush=True)
    print('Avergae valid pr for right kg:', np.mean(true_key_valid_pr))
    print('Avergae valid pr for wrong kg:', np.mean(wrong_key_valid_pr))
    if right_structure:
        print('For right kg: average score is {}, max score is {}, min score is {}'.format(np.mean(true_key_score), np.max(true_key_score), np.min(true_key_score)))
        print('For right kg: average key rank is {}, median key rank is {}'.format(np.mean(true_key_rank), np.median(true_key_rank)))
        print('For wrong kg: average score is {}, max score is {}, min score is {}'.format(np.mean(wrong_key_score), np.max(wrong_key_score), np.min(wrong_key_score)))
    else:
        print('For right kg: average score is {}, max score is {}, min score is {}'.format(np.mean(true_key_score), np.max(true_key_score), np.min(true_key_score)))
        print('For wrong kg: average score is {}, max score is {}, min score is {}'.format(np.mean(wrong_key_score), np.max(wrong_key_score), np.min(wrong_key_score)))

    print("Max succeed kg diff hw matric:", MAX_SUCCEED_KG_DIFF_HW)
    print("Key surviving time:", np.sum(sv_whether_surviving_kg))
    if right_structure:
        print("Attack success time:", np.sum(sv_whether_succeed))
    print("Average attack time: {} s".format(np.mean(sv_attack_time_consumption)))
    print("Average surviving key guess number:", np.mean(sv_num_surviving_kgs))
    if sv_best_kg_diff_hw:
        print("Average diff hw between true key and best key guess:", np.mean(sv_best_kg_diff_hw))
        print("Best kg diff hw show time:", sv_best_kg_diff_show_time)
    
    if not path.isdir(attack_saved_folder):
        makedirs(attack_saved_folder)
    np.save(attack_saved_folder + 'tk_rank.npy', np.array(true_key_rank, dtype=np.uint32))
    np.save(attack_saved_folder + 'tk_score.npy', np.array(true_key_score))
    np.save(attack_saved_folder + 'tk_valid_pr.npy', np.array(true_key_valid_pr))
    np.save(attack_saved_folder + 'wk_score.npy', wrong_key_score)
    np.save(attack_saved_folder + 'wk_valid_pr.npy', np.array(wrong_key_valid_pr))
    np.save(attack_saved_folder + 'sv_whether_surviving_kg.npy', sv_whether_surviving_kg)
    np.save(attack_saved_folder + 'sv_whether_succeed.npy', sv_whether_succeed)
    np.save(attack_saved_folder + 'sv_attack_time.npy', sv_attack_time_consumption)
    np.save(attack_saved_folder + 'sv_best_kg_diff_show_time.npy', sv_best_kg_diff_show_time)
    np.save(attack_saved_folder + 'sv_num_surviving_kgs.npy', sv_num_surviving_kgs)

if __name__ == '__main__':
    kg_low = 24
    m = 3
    dis_nr = 7
    diff = (0x40000000, 0)
    structure_size = 2**10
    net_path = 'saved_model/7/62_12-11-4-3-1-0_student_7_distinguisher.h5'
    selected_bits = [12,11,4,3,1,0]
    key_guess_bits = [28,27,26,25,24,4,3,2,1,0]
    c = 10
    attack_saved_folder = './attack_res/key_rank_test/'
    # Test key ranks and key scores using right plaintext structure
    test_key_rank(1000, net_path, selected_bits, dis_nr, diff, structure_size, kg_low, m, key_guess_bits, True, c, attack_saved_folder)
    # Test key ranks and key scores using wrong plaintext structure
    test_key_rank(1000, net_path, selected_bits, dis_nr, diff, structure_size, kg_low, m, key_guess_bits, False, c, attack_saved_folder+'bad_structure/')