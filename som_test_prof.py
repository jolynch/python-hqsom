'''
Created on Dec 4, 2011

@author: thilk

See: http://docs.python.org/library/profile.html
especially the part about pstats

Using pstats in a Python shell is best, e.g. ipython

'''

from som_test import *
import cProfile

a = "hqsom_77"
cProfile.run("eval(\"test_\"+a)()", "test_results.pstats")




