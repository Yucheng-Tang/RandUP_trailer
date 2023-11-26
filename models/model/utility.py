import numpy as np


class Utility:
    def __init__(self):
        pass

    @staticmethod
    def xl_2_numpy(worksheet):
        rows = worksheet.max_row
        cols = worksheet.max_column

        data = []
        for row in range(1, rows + 1):
            row_data = []
            for col in range(1, cols + 1):
                value = worksheet.cell(row=row, column=col).value
                row_data.append(value)
            data.append(row_data)

        np_data = np.array(data)
        return np_data

    @staticmethod
    def relative_distance(initial_states, obstacles):
        initial_position = initial_states[:, 0:2]
        obstacle = obstacles[:, :obstacles.shape[1] - 2].reshape(obstacles.shape[0], 2, -1)
        distance = obstacle - initial_position.reshape(initial_position.shape[0], 2, 1)
        return distance.reshape(distance.shape[0], -1)
