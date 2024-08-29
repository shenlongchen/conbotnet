# HLA	sequence	length	log_ic50
# DRA*01:01-DRB1*01:01	AAKPAAAATATATAA	15	0.291023
def read_txt_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            
            line = line.strip()
            peptide_seq, log_ic50, mhc = line.split('\t')
            if mhc[0:3] == 'HLA':
                mhc = mhc[4:8]+"*"+mhc[8:10]+":"+mhc[10:17]+"*"+mhc[17:19]+":"+mhc[19:21]
            elif mhc[0:3] == 'DRB':
                mhc = "DRA*01:01-"+mhc[0:4]+"*"+mhc[5:7]+":"+mhc[7:9]
            new_line = f'{mhc}\t{peptide_seq}\t{len(peptide_seq)}\t{log_ic50}'
            
            data_list.append(new_line)
    return data_list

def write_list_to_txt(data_list, file_path):
    with open(file_path, 'w') as file:
        file.write('mhc\tpeptide_seq\tlength\tlog_ic50\n')
        for item in data_list:
            file.write(str(item) + '\n')




file_path = 'code_and_dataset/dataset/data.txt'
txt_data = read_txt_file(file_path)
output_file_path = 'code_and_dataset/dataset/BD_2016_NEW.txt'  
write_list_to_txt(txt_data, output_file_path)


