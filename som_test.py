from som import *

tests = ("som","rsom")

input_vectors = np.array([
        [0.1 , 0.1 , 0.1 , 0.1],
        [.01 ,.001 , 0.6 , 0.8 ],
        [0.3 , 0.3 , 0.3 , 0.3],
        [0.0 , 0.8 , 0.0 , 0.0],
        [1.0 , 0.9 , 0.95, 0.82],
        [0.35,0.95 , 0.24, 0.76]])
rate, spread, size, input_size = .4, .2, len(input_vectors), len(input_vectors[0])

def test_som():
    som1 = SOM(input_size, size)
    assert som1
    
    #Test that a single vector can be trained on
    print "-- Training on single input --"
    for i in range(10):
        som1.update(input_vectors[0], rate, spread)
    print "Got MSE of {}".format(som1.mse(input_vectors[0]))
    assert som1.mse(input_vectors[0]) < 1e-3
    
    #Test that all vectors can be trained on in a 1:1 network input_size = size
    som1 = SOM(input_size, size)
    print "-- Training on all inputs --"
    for i in range(1000):
        som1.update(input_vectors[i%len(input_vectors)], rate, spread)
   
    total_mse = 0
    for inp in input_vectors:
        total_mse += som1.mse(inp)
        print "Got MSE of {}".format(som1.mse(inp))
        assert som1.mse(inp) < .3
    assert total_mse < .05 * len(input_vectors)
    #Checking Activation vectors
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

#I'm kind of unsure how to really test this ...
def test_rsom():
    rsom1 = RSOM(input_size, size)
    alpha = .3
    #Test a time dependent sequence 
    print "-- Training on alternating values --"
    for i in range(1000):
        rsom1.update(input_vectors[i%2], rate, spread, alpha)
    rsom1.update(input_vectors[2], rate, spread, alpha)
    for i in range(3):
        print "Got MSE of {}".format(rsom1.mse(input_vectors[i]))
        print "Activation vector: {}".format(rsom1.activation_vector(input_vectors[i], True))
        assert rsom1.mse(input_vectors[i%2]) < .3 

if __name__ == "__main__":
    for test in tests:
        print "#"*80
        print "Running test on: {}".format(test)
        print "-"*80
        try:
            eval("test_"+test)()
        except Exception as e :
            print e
            print "!!! ERROR !!!"
        else:
            print "SUCCESS"

        print "#"*80

