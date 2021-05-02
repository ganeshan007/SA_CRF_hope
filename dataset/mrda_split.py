# https://github.com/Franck-Dernoncourt/naacl2016

train_set_idx = ['Bdb001', 'Bed002', 'Bed004', 'Bed005', 'Bed008', 'Bed009', 'Bed011', 'Bed013', 'Bed014', 'Bed015', 'Bed017', 'Bmr002', 'Bmr003', 'Bmr006', 'Bmr007', 'Bmr008', 'Bmr009', 'Bmr011', 'Bmr012', 'Bmr015', 'Bmr016', 'Bmr020', 'Bmr021', 'Bmr023', 'Bmr025', 'Bmr026', 'Bmr027', 'Bmr029', 'Bmr031', 'Bns001', 'Bns002', 'Bns003', 'Bro003', 'Bro005', 'Bro007', 'Bro010', 'Bro012', 'Bro013', 'Bro015', 'Bro016', 'Bro017', 'Bro019', 'Bro022', 'Bro023', 'Bro025', 'Bro026', 'Bro028', 'Bsr001', 'Btr001', 'Btr002', 'Buw001']

valid_set_idx = ['Bed003', 'Bed010', 'Bmr005', 'Bmr014', 'Bmr019', 'Bmr024', 'Bmr030', 'Bro004', 'Bro011', 'Bro018', 'Bro024']

test_set_idx = ['Bed006', 'Bed012', 'Bed016', 'Bmr001', 'Bmr010', 'Bmr022', 'Bmr028', 'Bro008', 'Bro014', 'Bro021', 'Bro027']

assert len(train_set_idx + valid_set_idx + test_set_idx) == 73
# Not Included (as suggested by the data set creators): Bmr013  (Meetings Eval Dev Set), Bmr018  (Meetings Eval Dev Set).