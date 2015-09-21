[![Build Status](https://travis-ci.org/jolynch/python-hqsom.svg)](https://travis-ci.org/jolynch/python-hqsom)

python-hqsom
============

Python Implementation and improvements to the HQSOM deep learning algorithm.

Check out som.py, rsom.py, and hqsom\_qudio.py for the basic library.  There is
also a genetic algorithm included to help you select network topologies, see
audio\_learning and letter\_learning.py for examples.

Testing
=======
Most of the testing is based on replicating results presented by J.W. Miller
and P.H. Lommel in their paper: "Biomimetic sensory abstraction using
hierarchical quilted self-organizing maps"

To test run:

```python -m pytest tests```

