import re


def clean_miss_file(file_path):
    text = open(file_path).read()
    text = re.sub('INFO:.+', '', text)
    text = re.sub('.+(?=datas)', '', text)
    text = re.sub('\n\n', '\n', text)
    lines = text.split('\n')
    not_empty_lines = [line for line in lines if line.strip()]
    open(file_path, 'w').write('\n'.join(not_empty_lines))


clean_miss_file(r'/home/ira/Desktop/inception-trial/datas/UrbanSound8K/training results/wave forms/missclassified.txt')
