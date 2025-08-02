import os
import re

def is_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def clean_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned_lines = []
    for line in lines:
       
        if '#' in line:
            code, comment = line.split('#', 1)
            if is_chinese(comment):
                line = code.rstrip() + '\n'
        if not is_chinese(line):  
            cleaned_lines.append(line)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

def scan_directory(directory, extensions=('.py', '.sh', '.cpp', '.c', '.h', '.java', '.js', '.ipynb')):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                clean_file(os.path.join(root, file))


scan_directory('./')
