import csv

# I read csv and make it into a text file with only quotes and <|endoftext|>'s
with open('insparation.csv', newline='', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    outfile = open("quotescsvClean.txt", 'w', encoding="utf8")
    for row in reader:
        #print(row['Quote'])
        cleaned_quote = row['Quote'].replace('\n', ' ').replace('\r', ' ').strip()
        outfile.write("Quote: " + cleaned_quote + "<|endoftext|>" +"\n") # I forgot to add Quote: to the beginning an I now have to retrain my model
    outfile.close()

# NOTE TO SELF - I need to add a section for cleaning strings that contain weird characters from utf-8 encoding
# But I'm lazy rn so it probably on't happen until later when I hate past me
# Will probably do it when I am making my ai template