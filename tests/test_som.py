from common_data import *
from hqsom.som import SOM


def test_improved_som():
    use_pure = False
    som1 = SOM(input_size, size, pure=use_pure)
    assert som1

    # Test that a single vector can be trained on
    print "-- Training on single input --"
    for i in range(10):
        som1.update(input_vectors[0], rate, spread)
    print "Got MSE of {}".format(som1.mse(input_vectors[0]))
    assert som1.mse(input_vectors[0]) < 1e-3

    # Test that all vectors can be trained on in a 1:1 network input_size = size
    som1 = SOM(input_size, size, pure=use_pure)
    print "-- Training on all inputs --"
    for i in range(1000):
        som1.update(input_vectors[i%len(input_vectors)], rate, spread)

    total_mse = 0
    for inp in input_vectors:
        total_mse += som1.mse(inp)
        print "Got MSE of {}".format(som1.mse(inp))
        assert som1.mse(inp) < .3
    assert total_mse < .05 * len(input_vectors)
    # Checking Activation vectors
    activated = set()
    for inp in input_vectors:
        activated.add(som1.bmu(inp))
        print "Applying signal: {}".format(inp)
        print "Activating {}".format(som1.units[som1.bmu(inp)])
    err = abs(len(activated) - len(input_vectors))
    print activated
    print "All activated units: {}".format(activated)
    print "Error: {} vs max {}".format(err, .5*len(input_vectors))
    assert err <= .5*len(input_vectors)
