def count_string_in_large_file(filename, search_string):
    count = 0
    try:
        with open(filename, 'r') as file:
            for line in file:
                count += line.count(search_string)
        return count
    except FileNotFoundError:
        return f"Error: The file {filename} was not found."
'''
file_name = 'quotescsvClean.txt'
string_to_find = '<|endoftext|>'
occurrences = count_string_in_large_file(file_name, string_to_find)
print(f"The string '{string_to_find}' appears {occurrences} times in {file_name}.")

print(type(occurrences))
batch_size = 8
planned_epochs = 50
the_num = (occurrences / batch_size) * planned_epochs
print(round(the_num, -3)) 

print(int(round(the_num, -3)))
'''