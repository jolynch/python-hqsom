from common_data import *
from hqsom.hqsom import HQSOM


use_pure = False


def test_hqsom():
    """ Tests the hqsom algorithm on horizontal and vertical lines and ensures
    that the hqsom can properly differentiate the two patterns """
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])

    g1,g2,s1,s2,a = .1,.1,16,90,.1
    hqsom = HQSOM(9,18,3, use_pure_implementation=use_pure)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)
    num_cycles, num_repeats = 25, 11
    total_run_count, seq_count = num_cycles*num_repeats*9, 0 
    for j in range(num_cycles):
        for i in range(num_repeats):
            print "update {}/{}".format(seq_count,total_run_count)
            hqsom.reset()
            seq = ()
            if i %2 == 0:
                seq = (1,2,3,1,2,3)
            else:
                seq = (4,5,6,4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            hqsom.reset()
            seq_count += 9

    c = [hqsom.activation_vector(t) for t in test_data]
    print c
    assert c[0] != c[1] and c[1] != c[4]
    assert c[1] == c[2] and c[2] == c[3]
    assert c[4] == c[5] and c[5] == c[6]
    assert c[3] != c[4]


def test_hqsom_noise(noise_std=.1):
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])
    # Add in gausian noise
    noise = np.random.normal(0.0,noise_std,test_data.shape)
    test_data = test_data + noise 
    g1,g2,s1,s2,a = .1,.1,16,90,.1
    # Due to the noise we have to add many more map units
    hqsom = HQSOM(9,18,3, use_pure_implementation=use_pure)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)
    num_cycles, num_repeats = 25, 11
    total_run_count, seq_count = num_cycles*num_repeats*9, 0 
    for j in range(num_cycles):
        for i in range(num_repeats):
            print "update {}/{}".format(seq_count,total_run_count)
            hqsom.reset()
            if i %2 == 0:
                seq = (1,2,3,1,2,3)
            else:
                seq = (4,5,6,4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            hqsom.reset()
            seq_count += 9

    # Re-do the test data to test on different noisy data
    print "Generating different test data for activating"
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])
    # Add in gausian noise
    noise = np.random.normal(0.0,noise_std,test_data.shape)
    test_data = test_data + noise
    g1,g2,s1,s2,a = .1,.1,16,90,.1

    c = [hqsom.activation_vector(t) for t in test_data]
    print c
    assert c[0] != c[1] and c[1] != c[4]
    assert c[1] == c[2] and c[2] == c[3]
    assert c[4] == c[5] and c[5] == c[6]
    assert c[3] != c[4]

def xtest_hqsom_noise_multiple():
    num_errors, num_tests, noise_std = 0, 100, .2
    for i in range(num_tests):
        try:
            test_hqsom_noise(noise_std)
        except:
            num_errors += 1
    print "Passed {} out of {}".format(num_tests-num_errors, num_tests)
    assert num_errors < .25 * num_tests





