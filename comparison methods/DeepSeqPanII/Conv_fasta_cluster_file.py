def fasta_to_single_line(fasta_string):
    lines = fasta_string.strip().split("\n")
    single_line_sequences = []
    current_name = ""
    current_sequence = ""

    for line in lines:
        if line.startswith(">"):
            if current_name and current_sequence:
                fasta_format_sequence = f"{current_name}\t{current_sequence}"
                single_line_sequences.append(fasta_format_sequence)
                current_sequence = ""
            current_name = line[1:]
        else:
            current_sequence += line.strip()

    if current_name and current_sequence:
        fasta_format_sequence = f"{current_name}\t{current_sequence}"
        single_line_sequences.append(fasta_format_sequence)

    return "\n".join(single_line_sequences)


def read_fasta_file(input_file):
    with open(input_file, "r") as file:
        fasta_string = file.read()
    return fasta_string


def write_single_line_sequences(output_file, single_line_sequences):
    with open(output_file, "w") as file:
        file.write(single_line_sequences)


# input_file = 'code_and_dataset/dataset/CLUATAL_OMEGA_A_NEW.txt' 
# output_file = 'code_and_dataset/dataset/CLUATAL_OMEGA_A_NEW_alligned.txt'  

input_file = 'code_and_dataset/dataset/CLUATAL_OMEGA_B_NEW.txt' 
output_file = 'code_and_dataset/dataset/CLUATAL_OMEGA_B_NEW_alligned.txt'

fasta_string = read_fasta_file(input_file)
single_line_sequences = fasta_to_single_line(fasta_string)
write_single_line_sequences(output_file, single_line_sequences)
