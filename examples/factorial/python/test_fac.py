import unittest
from fac import factorial # This line added by me!

class TestFactorial(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(factorial(5), 120, 'factorial(5)')
        self.assertEqual(factorial(1), 1, 'factorial(1)')
        self.assertEqual(factorial(0), 1, 'factorial(0)')

# Run the unit tests        
unittest.main(argv=[''], exit=False)
