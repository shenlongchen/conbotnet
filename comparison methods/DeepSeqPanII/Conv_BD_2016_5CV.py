# HLA	sequence	length	log_ic50
# DRA*01:01-DRB1*01:01	AAKPAAAATATATAA	15	0.291023
def read_txt_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:

            line = line.strip()

            data_list.append(line)
    return data_list

def write_list_to_txt(data_list, cv_list, file_path):
    with open(file_path, 'w') as file:
        file.write('mhc\tpeptide_seq\tlength\tlog_ic50\tcv_id\n')
        index = 0
        for item in data_list[1:]:
            file.write(str(item) + '\t'+cv_list[index]+'\n')
            index += 1





cv_file_path = 'code_and_dataset/dataset/cv_id.txt'
cv_data = read_txt_file(cv_file_path)
file_path = 'code_and_dataset/dataset/BD_2016_NEW.txt'
txt_data = read_txt_file(file_path)
output_file_path = 'code_and_dataset/dataset/BD_2016_5cv.txt'
write_list_to_txt(txt_data, cv_data, output_file_path)


