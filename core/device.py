# -*- coding: utf-8 -*-

"""TODO"""


class Device:
    """TODO"""
    unique_id_count = 0

    def __init__(self):
        """TODO"""
        self.active = True
        self.unique_id = Device.unique_id_count
        Device.unique_id_count += 1
        self.name = "Device" + str(self.unique_id)

    def evolve(self):
        """TODO"""
        pass

    def execute(self):
        """TODO"""
        pass

    def flush(self):
        """TODO"""
