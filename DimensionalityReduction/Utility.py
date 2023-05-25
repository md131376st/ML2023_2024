def VectorCol(data):
    return data.reshape((data.size, 1))


def VectorRow(data):
    return data.reshape((1, data.size))