import numpy as np


class Expression(object):
    '''Base expression class that represents anything in our computational
    graph. Everything should be one of these.'''
    def __init__(self, grad=False):
        self.grad = grad

    def eval(self, feed_dict):
        '''Evaluates the entire computation graph given a dictionary of
        variables mapped to values.'''
        return self._eval(feed_dict, dict())
    
    def _eval(self, feed_dict, cache_dict):
        '''Helper - Evaluates the computation graph recursively.'''
        raise NotImplementedError

    def d(self, feed_dict):
        '''Evaluates the derivative at the points given, returns to user'''
        return self._d(feed_dict, dict())

    def _d(self):
        '''Helper - Evaluates the differentiation products recursively.'''
        raise NotImplementedError

    def __add__(self, other):
        try:
            # Propagate the need for gradient if one thing needs gradient
            return Addition(self, other, grad=(self.grad and other.grad))
        except AttributeError:
            return Addition(self, Constant(other), grad=self.grad)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            return Multiplication(self, other, grad=(self.grad and other.grad))
        except AttributeError:
            return Multiplication(self, Constant(other), grad=self.grad)
    
    def __rmul__(self, other):
        # TODO: Multiplication not commutative if we enable matrix support
        return self.__mul__(other)

class Variable(Expression):
    def __init__(self, name, grad=True):
        self.grad = grad
        self.name = name
    
    def _eval(self, feed_dict, cache_dict):
        # Check if the user specified either the object in feed_dict or
        # the name of the object in feed_dict
        if self in feed_dict:
            return feed_dict[self]
        elif self.name in feed_dict:
            return feed_dict[self.name]
        else:
            raise ValueError('Unbound variable %s' % self.name)


class Constant(Expression):
    '''Represents a constant.'''
    def __init__(self, val, grad=False):
        super().__init__(grad=grad)
        self.val = val
    
    def _eval(self, feed_dict, cache_dict):
        return self.val


class Binop(Expression):
    '''Utilities common to all binary operations in the form Op(a, b)'''
    def __init__(self, expr1, expr2, grad=False):
        super().__init__(grad=grad)
        self.expr1 = expr1
        self.expr2 = expr2


class Addition(Binop):
    '''Addition, in the form A + B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 + res2
        return cache_dict[id(self)]


class Subtraction(Binop):
    '''Subtraction, in the form A - B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 - res2
        return cache_dict[id(self)]


class Multiplication(Binop):
    '''Multiplication, in the form A * B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 * res2
        return cache_dict[id(self)]


class Division(Binop):
    '''Division, in the form A / B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 / res2
        return cache_dict[id(self)]

