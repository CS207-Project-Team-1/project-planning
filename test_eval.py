'''Testing suite for testing only the evaluation part of the script, with no
automatic differentiation.'''
import unittest
import ad

class TestAddition(unittest.TestCase):
    pass


class TestSubtraction(unittest.TestCase):

    def test_constants(self):
        c1 = ad.Constant(1.0)
        c2 = ad.Constant(5.0)
        y = c1 - c2
        self.assertEqual(-4.0, y.eval({}))
    
    def test_casted_constants_left(self):
        c1 = ad.Constant(1.0)
        self.assertEqual((c1 - 5).eval({}), -4.0)
        self.assertEqual((c1 - 1).eval({}), 0)
        self.assertEqual((c1 - (-5)).eval({}), 6.0)

    def test_casted_constants_right(self):
        c1 = ad.Constant(1.0)
        self.assertEqual((5 - c1).eval({}), 4.0)
        self.assertEqual((1 - c1).eval({}), 0)
        self.assertEqual(((-5) - c1).eval({}), -6.0)

if __name__ == '__main__':
    unittest.main()