from common_data import *
from hqsom.rsom import RSOM

def test_rsom():
    use_pure = False
    rsom1 = RSOM(input_size, size, pure=use_pure)
    alpha = .3
    # Test a time dependent sequence
    print "-- Training on alternating values --"
    for i in range(1000):
        rsom1.update(input_vectors[i%2], rate, spread, alpha)
    rsom1.update(input_vectors[2], rate, spread, alpha)
    rsom1.reset()
    assert abs(rsom1.differences[0][0] - 0) < 0.01
    for i in range(3):
        print "Got MSE of {}".format(rsom1.mse(input_vectors[i]))
        print "Activation vector: {}".format(rsom1.activation_vector(input_vectors[i], True))
        assert rsom1.mse(input_vectors[i%2]) < .3
