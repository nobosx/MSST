import numpy as np
import speck as sp
from keras.models import load_model

def cal_hw(input_int, length=32):
    hw_res = 0
    for i in range(length):
        hw_res = hw_res + (input_int & 1)
        input_int = input_int >> 1
    return hw_res

# Convert bit array to int
# X.shape = (length, :)
def bits_to_int(X, length):
    res = 0
    for i in range(length):
        res = (res << 1) | (X[i])

# Convert int to bit array
# x.shape = (length, :)
# If reverse = True, X[0] is the most significant bit. Otherwise X[0] is the least significant bit.
def int_to_bits(x, length, reverse=True):
    X = []
    bit_indexes = range(length)
    if reverse:
        bit_indexes = reversed(bit_indexes)
    for i in bit_indexes:
        X.append((x >> i) & 1)
    X = np.array(X, dtype=np.uint8)
    return X

def extract_sensitive_bits(raw_x, bits):
    # get new-x according to sensitive bits
    id0 = [sp.WORD_SIZE() - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + sp.WORD_SIZE() * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]
    # print('new_x shape is ', np.shape(new_x))

def extract_selected_bits_int(input, selected_bits, embedding=True):
    if not isinstance(input, np.ndarray):
        input = int(input)
    res = 0
    if embedding:
        for i in selected_bits:
            res = (res << 1) | ((input >> i) & 1)
    else:
        for i in selected_bits:
            res |= input & (1 << i)
    return res

# Extract selected bits of every integer in int_array and combine them into one integer
def extract_bits_and_combine(int_array, selected_bits):
    assert len(int_array) * len(selected_bits) <= 64
    res = 0
    single_length = len(selected_bits)
    for int_entry in int_array:
        res = (res << single_length) | extract_selected_bits_int(int_entry, selected_bits, True)
    return res

# Combine all the integers in int_array into one integer
def combine_ints(int_array, int_lengths):
    total_length = 0
    for i in int_lengths:
        total_length += i
    assert total_length <= 64
    res = 0
    for i in range(len(int_array)):
        if not isinstance(int_array[i], np.ndarray):
            res = (res << int_lengths[i]) | int(int_array[i])
        else:
            res = (res << int_lengths[i]) | int_array[i]
    return res

def gen_lookup_table_from_net(net_path, selected_bits, saved_path):
    net = load_model(net_path)
    input_length = 4 * len(selected_bits)
    input_int = np.arange(2**input_length, dtype=np.uint64)
    input_bits = int_to_bits(input_int, input_length, True)
    input_bits = np.transpose(input_bits)
    Z = net.predict(input_bits, batch_size=10000, verbose=0).flatten()
    Z = np.log2(Z / (1 - Z))
    np.save(saved_path, Z)

if __name__ == "__main__":
    # 9-round distinguisher
    diff_pos = 66
    selected_bits = [21, 20]
    net_path = 'saved_model/9/21-20_{}_student_distinguisher.h5'.format(diff_pos)
    saved_path = 'lookup_table/9/21-20_{}_student_distinguisher.npy'.format(diff_pos)
    gen_lookup_table_from_net(net_path, selected_bits, saved_path)