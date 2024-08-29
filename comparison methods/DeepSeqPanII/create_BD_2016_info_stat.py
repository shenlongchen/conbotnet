import math

CUTOFF = 1.0 - math.log(500, 50000)
# HLA	sequence	length	log_ic50
# DRA*01:01-DRB1*01:01	AAKPAAAATATATAA	15	0.291023
dict_mhc_count = {}
dict_mhc_pos_count = {}
mhc_set = set()
def read_txt_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符
            line = line.strip()
            data_list.append(line)
    return data_list

def write_list_to_txt(data_list, file_path):
    with open(file_path, 'w') as file:
        # allele, samples
        file.write('allele\tsamples\n')
        for item in data_list:
            file.write(str(item) + '\n')




file_path = 'code_and_dataset/dataset/BD_2016_NEW.txt'
txt_data = read_txt_file(file_path)
for item in txt_data[1:]:
    mhc = item.split('\t')[0]
    mhc_set.add(mhc)
    log_ic50 = item.split('\t')[3]

    if mhc not in dict_mhc_count:
        dict_mhc_count[mhc] = 1
    else:
        dict_mhc_count[mhc] += 1

    if float(log_ic50) >= CUTOFF:
        if mhc not in dict_mhc_pos_count:
            dict_mhc_pos_count[mhc] = 1
        else:
            dict_mhc_pos_count[mhc] += 1

txt_data = []
for mhc in mhc_set:
    if mhc in dict_mhc_count and mhc in dict_mhc_pos_count:
        if dict_mhc_count[mhc] > 30 and dict_mhc_pos_count[mhc] >= 3:
            line_mhc_count = mhc + '\t' + str(dict_mhc_count[mhc])
            txt_data.append(line_mhc_count)
txt_data.sort()
output_file_path = 'code_and_dataset/dataset/BD2016_allele_info_stat_new.txt'  
write_list_to_txt(txt_data, output_file_path)


