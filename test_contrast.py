
from skynet import SkyNet
from PyQt4 import QtGui, QtCore
from random import randrange
from sys import argv, exit
from random import randrange

def rand_bw():
    return "white" if randrange(0, 2) else "black"

def rand_color():
    return randrange(0, 256), randrange(0, 256), randrange(0, 256)

def normalize(color):
    return map(lambda x: x / 255., color)

def brightness(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

class ColorTeacher(object):

    def __init__(self, network):
        self.network = network

    def train(self, n=1000, verbose=False):
        """ Train the neural network by feeding the network and apply the back
            propagation. The learning rate is represented by `eta` while
            `alpha` is the momentum factor.
        """
        if verbose:
            print("\nTrain the network over %d iterations:\n" % n)
        for i in range(n):
            color = rand_color()
            self.network.feed_forward(color)
            error = self.network.back_propagate(
                [1. - brightness(normalize(color))]
                )
            if verbose and i % (n / 10) == 0:
                print("  Error: %.8f" % error)

class Window(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.net = SkyNet(
            nb_input=3, nb_hidden=2, nb_output=1,
            eta=0.5, alpha=0.1,
            )

        teacher = ColorTeacher(self.net)
        teacher.train(100000, verbose=True)

        layout = QtGui.QVBoxLayout(self)

        buttons_layout = QtGui.QHBoxLayout()
        layout.addLayout(buttons_layout)

        color = rand_color()
        self.inputs = normalize(color)
        self.black_button = QtGui.QPushButton('BLACK', self)
        self.black_button.clicked.connect(self.on_black)
        self.update_button_style('black', color)
        buttons_layout.addWidget(self.black_button)

        self.white_button = QtGui.QPushButton('WHITE', self)
        self.white_button.clicked.connect(self.on_white)
        self.update_button_style('white', color)
        buttons_layout.addWidget(self.white_button)

        self.auto_button = QtGui.QPushButton("BRAIN (error %.5f)" % 0.0, self)
        self.auto_button.clicked.connect(self.on_auto)
        layout.addWidget(self.auto_button)

        self.label = QtGui.QLabel("Result: 0, Choice: white")
        layout.addWidget(self.label)

    def update_button_style(self, text, background):
        button = getattr(self, text + "_button")
        button.setStyleSheet("""
            background-color: rgb%s;
            color: %s;
            border: none;
            padding: 50px;
        """ % (background, text))

    def train(self, inputs, targets):
        self.net.feed_forward(inputs)
        error = self.net.back_propagate(targets)
        self.auto_button.setText("BRAIN (error %.5f)" % error)

    def on_black(self):
        color = rand_color()
        self.inputs = normalize(color)
        self.update_button_style('black', color)
        self.update_button_style('white', color)
        self.train(self.inputs, [1.0])
        
    def on_white(self):
        color = rand_color()
        self.inputs = normalize(color)
        self.update_button_style('black', color)
        self.update_button_style('white', color)
        self.train(self.inputs, [0.0])

    def on_auto(self):
        result = self.net.feed_forward(self.inputs)[0]
        self.label.setText("Result: %.2f, Choice: %s, Brighness: %.2f" % (
            result, ("white" if result < 0.5 else "black"),
            brightness(self.inputs),
            ))

if __name__ == '__main__':
    app = QtGui.QApplication(argv)
    window = Window()
    window.show()
    exit(app.exec_())
