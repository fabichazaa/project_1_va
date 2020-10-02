def label_to_int(label):
    if label == '5-point-star':
        return 1
    elif label == 'rectangle':
        return 2
    elif label == 'triangle':
        return 3
    elif label == 'circle':
        return 4
    else:
        raise Exception('unknown class_label')


def int_to_label(label):
    if label == 1:
        return '5-point-star'
    elif label == 2:
        return 'rectangle'
    elif label == 3:
        return 'triangle'
    elif label == 4:
        return 'circle'
    else:
        raise Exception('unknown class_label')
