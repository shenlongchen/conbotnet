def normalize_protein_sequence(sequence, description):

    normalized_sequence = ''.join(character for character in sequence if character.isalpha())

 
    fasta_format_sequence = f">{description}\n{normalized_sequence}"

    return fasta_format_sequence

def read_txt_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:

            line = line.strip()

            data_list.append(line)
    return data_list


# file_path = 'code_and_dataset/dataset/CLUATAL_OMEGA_A_chains_aligned_FLATTEN_all.txt'
file_path = 'code_and_dataset/dataset/CLUATAL_OMEGA_B_chains_aligned_FLATTEN_all.txt'
txt_data = read_txt_file(file_path)


my_list = []
for line in txt_data[1:]:

    line_data = line.split('\t')

    protein_sequence = line_data[1]
    description = line_data[0]


    fasta_format_sequence = normalize_protein_sequence(protein_sequence, description)
    my_list.append(fasta_format_sequence)

def write_list_to_txt(data_list, file_path):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(str(item) + '\n')


output_file_path = file_path.replace('.txt', '_fasta.txt')
write_list_to_txt(my_list, output_file_path)
