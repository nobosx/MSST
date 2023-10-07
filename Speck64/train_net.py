import student_net as sn
import speck as sp

# Train a 7-round neural network distinguisher 
diff_pos = 62
selected_bits = [12,11,4,3,1,0]
nr = 7
diff_l, diff_r = 0, 0
if diff_pos < sp.WORD_SIZE():
    diff_r = 1 << diff_pos
else:
    diff_l = 1 << (diff_pos - sp.WORD_SIZE())
diff = (diff_l, diff_r)
saved_folder = './saved_model/7/12-11-4-3-1-0_{}_'.format(diff_pos)
sn.train_speck_distinguisher(10, nr, 1, diff, selected_bits, saved_folder)

# Train a 6-round neural netwrok distinguisher
diff_pos = 42
selected_bits = [21, 20]
nr = 6
diff_l, diff_r = 0, 0
if diff_pos < sp.WORD_SIZE():
    diff_r = 1 << diff_pos
else:
    diff_l = 1 << (diff_pos - sp.WORD_SIZE())
diff = (diff_l, diff_r)
saved_folder = './saved_model/6/21-20_{}_'.format(diff_pos)
sn.train_speck_distinguisher(10, nr, 1, diff, selected_bits, saved_folder)