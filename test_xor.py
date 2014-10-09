
from skynet import SkyNet
from random import seed

seed(0)

class PatternTeacher(object):

    def __init__(self, network, patterns):
        self.network = network
        self.patterns = patterns

    def train(self, n=1000, verbose=False):
        """ Train the neural network by feeding the network and apply the back
            propagation. The learning rate is represented by `eta` while
            `alpha` is the momentum factor.
        """
        if verbose:
            print("\nTrain the network over %d iterations:\n" % n)
        for i in range(n):
            error = 0.0
            for inputs, targets in self.patterns:
                self.network.feed_forward(inputs)
                error += self.network.back_propagate(targets)
            if verbose and i % (n / 10) == 0:
                print("  Error: %.8f" % error)

def test_xor():

    patterns = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    net = SkyNet(
        nb_input=2, nb_hidden=4, nb_output=1,
        eta=0.3, alpha=0.01,
        )
    teacher = PatternTeacher(net, patterns)
    teacher.train(n=10000, verbose=True)
    print("\nTest the trained network:\n")
    for inputs, targets in patterns:
        print("  Input:\t%s" % inputs)
        print("  Expected:\t%s" % targets)
        print("  Output:\t%s" % net.feed_forward(inputs))
        print("")

if __name__ == '__main__':
    test_xor()
