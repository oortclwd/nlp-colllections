def read_text(fn):
    """
    :param fn: 파일 경로
    파일의 각 줄은 "positive    문장1" 형태여야 한다.
    """
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':

                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts

def check_int(label):
    try:
        int(label)
        return True
    except ValueError as e:
        return False