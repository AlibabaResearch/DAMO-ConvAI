# eos tokens definition
eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'qspn': '<eos_q>', 'qspn_gen': '<eos_q>', 'pv_qspn': '<eos_q>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}

# sos tokens definition
sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'qspn': '<sos_q>', 'qspn_gen': '<sos_q>', 'pv_qspn': '<sos_q>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>'}

# db tokens definition
db_tokens = ['<sos_db>', '<eos_db>',
             '[book_nores]', '[book_fail]', '[book_success]',
             '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']


# understand tokens definition
def get_understand_tokens(prompt_num_for_understand):
    understand_tokens = []
    for i in range(prompt_num_for_understand):
        understand_tokens.append(f'<understand_{i}>')
    return understand_tokens


# all special tokens definition
def get_special_tokens(understand_tokens):
    special_tokens = ['<go_r>', '<go_b>', '<go_a>', '<go_d>',
                      '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<eos_d>', '<eos_q>',
                      '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', '<sos_q>'] \
                     + db_tokens \
                     + understand_tokens
    return special_tokens
