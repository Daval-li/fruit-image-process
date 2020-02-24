# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:38:53 2020

@author: han
"""

from PyQt5.Qt import *
import sys
from CNNFruit import CNN_init,run_CNN
from UISet import *

if __name__ == '__main__':
    dict_data,sess= CNN_init()
    app = QApplication(sys.argv)

    window = MyWindow(dict_data,sess)
    window.show()

    sys.exit(app.exec())



