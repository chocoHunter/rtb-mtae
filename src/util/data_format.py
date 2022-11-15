class EncodeData:
    def __init__(self, X, Y, Z, B):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.B = B


class CensoredBatchData:
    def __init__(self, data, batch_id=-1, win_flag=True):
        self.batch_id = batch_id
        self.data = data
        self.win_flag = win_flag
        self.size = len(data.Z)