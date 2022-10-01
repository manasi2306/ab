"""Provides some utilities widely used by other modules"""
import bisect
import collections
import collections
import heapq
import operator
import os.path
import random
import math
import functools
from itertools import chain, combinations
# ______________________________________________________________________________
# Functions on Sequences and Iterables
def sequence(iterable):
	"""Coerce iterable to sequence, if it is not already one."""
	return (iterable if isinstance(iterable, collections.abc.Sequence)
		else tuple(iterable))
def removeall(item, seq):
	"""Return a copy of seq (or string) with all occurrences of item removed."""
	if isinstance(seq, str):
		return seq.replace(item, '')
	else:
		return [x for x in seq if x != item]
def unique(seq): # TODO: replace with set
	return list(set(seq))
def count(seq):
	return sum(bool(x) for x in seq)
def product(numbers):
	result = 1
	for x in numbers:
		result *= x
	return result
def first(iterable, default=None):
	try:
		return iterable[0]
	except IndexError:
		return default
	except TypeError:
		return next(iterable, default)
def is_in(elt, seq):
	return any(x is elt for x in seq)
def mode(data):
	[(item, count)] = collections.Counter(data).most_common(1)
	return item
def powerset(iterable):
	s = list(iterable)
	return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]
identity = lambda x: x
argmin = min
argmax = max
def argmin_random_tie(seq, key=identity):
	return argmin(shuffled(seq), key=key)
def argmax_random_tie(seq, key=identity):
	return argmax(shuffled(seq), key=key)
def shuffled(iterable):
	items = list(iterable)
	random.shuffle(items)
	return items
def histogram(values, mode=0, bin_function=None):
	if bin_function:
		values = map(bin_function, values)
		bins = {}
		for val in values:
			bins[val] = bins.get(val, 0) + 1
			if mode:
				return sorted(list(bins.items()), key=lambda x: (x[1], x[0]),
				reverse=True)
			else:
				return sorted(bins.items())
def dotproduct(X, Y):
	return sum(x * y for x, y in zip(X, Y))
def element_wise_product(X, Y):
	assert len(X) == len(Y)
	return [x * y for x, y in zip(X, Y)]
def matrix_multiplication(X_M, *Y_M):
	def _mat_mult(X_M, Y_M):
		assert len(X_M[0]) == len(Y_M)
		result = [[0 for i in range(len(Y_M[0]))] for j in range(len(X_M))]
		for i in range(len(X_M)):
			for j in range(len(Y_M[0])):
				for k in range(len(Y_M)):
					result[i][j] += X_M[i][k] * Y_M[k][j]
		return result

	result = X_M
	for Y in Y_M:
		result = _mat_mult(result, Y)
		return result
def vector_to_diagonal(v):
	diag_matrix = [[0 for i in range(len(v))] for j in range(len(v))]
	for i in range(len(v)):
		diag_matrix[i][i] = v[i]
	return diag_matrix
def vector_add(a, b):
	return tuple(map(operator.add, a, b))
def scalar_vector_product(X, Y):
	return [X * y for y in Y]
def scalar_matrix_product(X, Y):
	return [scalar_vector_product(X, y) for y in Y]
def inverse_matrix(X):
	assert len(X) == 2
	assert len(X[0]) == 2
	det = X[0][0] * X[1][1] - X[0][1] * X[1][0]
	assert det != 0
	inv_mat = scalar_matrix_product(1.0 / det, [[X[1][1], -X[0][1]], [-X[1][0], X[0][0]]])
	return inv_mat
def probability(p):
	return p > random.uniform(0.0, 1.0)
def weighted_sample_with_replacement(n, seq, weights):
	sample = weighted_sampler(seq, weights)
	return [sample() for _ in range(n)]
def weighted_sampler(seq, weights):
	totals = []
	for w in weights:
		totals.append(w + totals[-1] if totals else w)
	return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]
def rounder(numbers, d=4):
	if isinstance(numbers, (int, float)):
		return round(numbers, d)
	else:
		constructor = type(numbers)
		return constructor(rounder(n, d) for n in numbers)
def num_or_str(x):
	try:
		return int(x)
	except ValueError:
		try:
			return float(x)
		except ValueError:
			return str(x).strip()
def normalize(dist):
	if isinstance(dist, dict):
		total = sum(dist.values())
		for key in dist:
			dist[key] = dist[key] / total
			assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
		return dist
	total = sum(dist)
	return [(n / total) for n in dist]
def norm(X, n=2):
	return sum([x ** n for x in X]) ** (1 / n)
def clip(x, lowest, highest):
	return max(lowest, min(x, highest))
def sigmoid_derivative(value):
	return value * (1 - value)
def sigmoid(x):
	return 1 / (1 + math.exp(-x))
def step(x):
	return 1 if x >= 0 else 0
def gaussian(mean, st_dev, x):
	return 1 / (math.sqrt(2 * math.pi) * st_dev) * math.e ** (-0.5 * (float(x - mean) / st_dev) ** 2)
try:
	from math import isclose
except ImportError:
	def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
		return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
def weighted_choice(choices):
	total = sum(w for _, w in choices)
	r = random.uniform(0, total)
	upto = 0
	for c, w in choices:
		if upto + w >= r:
			return c, w
		upto += w

orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)
def turn_heading(heading, inc, headings=orientations):
	return headings[(headings.index(heading) + inc) % len(headings)]
def turn_right(heading):
	return turn_heading(heading, RIGHT)
def turn_left(heading):
	return turn_heading(heading, LEFT)
def distance(a, b):
	xA, yA = a
	xB, yB = b
	return math.hypot((xA - xB), (yA - yB))
def distance_squared(a, b):
	xA, yA = a
	xB, yB = b
	return (xA - xB) ** 2 + (yA - yB) ** 2
def vector_clip(vector, lowest, highest):
	return type(vector)(map(clip, vector, lowest, highest))

class injection():
	def __init__(self, **kwds):
		self.new = kwds
	def __enter__(self):
		self.old = {v: globals()[v] for v in self.new}
		globals().update(self.new)
	def __exit__(self, type, value, traceback):
		globals().update(self.old)
def memoize(fn, slot=None, maxsize=32):
	if slot:
		def memoized_fn(obj, *args):
			if hasattr(obj, slot):
				return getattr(obj, slot)
			else:
				val = fn(obj, *args)
				setattr(obj, slot, val)
				return val
	else:
		@functools.lru_cache(maxsize=maxsize)
		def memoized_fn(*args):
			return fn(*args)
	return memoized_fn

def name(obj):
	return (getattr(obj, 'name', 0) or getattr(obj, '__name__', 0) or getattr(getattr(obj, '__class__', 0), '__name__', 0) or str(obj))
def isnumber(x):
	return hasattr(x, '__int__')
def issequence(x):
	return isinstance(x, collections.abc.Sequence)
def print_table(table, header=None, sep=' ', numfmt='{}'):
	justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]
	if header:
		table.insert(0, header)
	table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]
	sizes = list(
		map(lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))))
	for row in table:
		print(sep.join(getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))
def open_data(name, mode='r'):
	aima_root = os.path.dirname(__file__)
	aima_file = os.path.join(aima_root, *['aima-data', name])
	return open(aima_file, mode=mode)
def failure_test(algorithm, tests):
	from statistics import mean
	return mean(int(algorithm(x) != y) for x, y in tests)
class Expr(object):
	def __init__(self, op, *args):
		self.op = str(op)
		self.args = args
	def __neg__(self):
		return Expr('-', self)
	def __pos__(self):
		return Expr('+', self)
	def __invert__(self):
		return Expr('~', self)
	def __add__(self, rhs):
		return Expr('+', self, rhs)
	def __sub__(self, rhs):
		return Expr('-', self, rhs)
	def __mul__(self, rhs):
		return Expr('*', self, rhs)
	def __pow__(self, rhs):
		return Expr('**', self, rhs)
	def __mod__(self, rhs):
		return Expr('%', self, rhs)
	def __and__(self, rhs):
		return Expr('&', self, rhs)
	def __xor__(self, rhs):
		return Expr('^', self, rhs)
	def __rshift__(self, rhs):
		return Expr('>>', self, rhs)
	def __lshift__(self, rhs):
		return Expr('<<', self, rhs)
	def __truediv__(self, rhs):
		return Expr('/', self, rhs)
	def __floordiv__(self, rhs):
		return Expr('//', self, rhs)
	def __matmul__(self, rhs):
		return Expr('@', self, rhs)
	def __or__(self, rhs):
		if isinstance(rhs, Expression):
			return Expr('|', self, rhs)
		else:
			return PartialExpr(rhs, self)
	def __radd__(self, lhs):
		return Expr('+', lhs, self)
	def __rsub__(self, lhs):
		return Expr('-', lhs, self)
	def __rmul__(self, lhs):
		return Expr('*', lhs, self)
	def __rdiv__(self, lhs):
		return Expr('/', lhs, self)
	def __rpow__(self, lhs):
		return Expr('**', lhs, self)
	def __rmod__(self, lhs):
		return Expr('%', lhs, self)
	def __rand__(self, lhs):
		return Expr('&', lhs, self)
	def __rxor__(self, lhs):
		return Expr('^', lhs, self)
	def __ror__(self, lhs):
		return Expr('|', lhs, self)
	def __rrshift__(self, lhs):
		return Expr('>>', lhs, self)
	def __rlshift__(self, lhs):
		return Expr('<<', lhs, self)
	def __rtruediv__(self, lhs):
		return Expr('/', lhs, self)
	def __rfloordiv__(self, lhs):
		return Expr('//', lhs, self)
	def __rmatmul__(self, lhs):
		return Expr('@', lhs, self)
	def __call__(self, *args):
		"Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."
		if self.args:
			raise ValueError('can only do a call for a Symbol, not an Expr')
		else:
			return Expr(self.op, *args)
# Equality and repr
	def __eq__(self, other):
		"'x == y' evaluates to True or False; does not build an Expr."
		return (isinstance(other, Expr) and self.op == other.op and self.args == other.args)
	def __hash__(self):
		return hash(self.op) ^ hash(self.args)
	def __repr__(self):
		op = self.op
		args = [str(arg) for arg in self.args]
		if op.isidentifier(): # f(x) or f(x, y)
			return '{}({})'.format(op, ', '.join(args)) if args else op
		elif len(args) == 1: # -x or -(x + 1)
			return op + args[0]
		else: # (x - y)
			opp = (' ' + op + ' ')
			return '(' + opp.join(args) + ')'
# An 'Expression' is either an Expr or a Number.
# Symbol is not an explicit type; it is any Expr with 0 args.
Number = (int, float, complex)
Expression = (Expr, Number)
def Symbol(name):
	return Expr(name)
def symbols(names):
	return tuple(Symbol(name) for name in names.replace(',', ' ').split())
def subexpressions(x):
	yield x
	if isinstance(x, Expr):
		for arg in x.args:
			yield subexpressions(arg)
def arity(expression):
	if isinstance(expression, Expr):
		return len(expression.args)
	else: # expression is a number
		return 0
class PartialExpr:
	def __init__(self, op, lhs):
		self.op, self.lhs = op, lhs
	def __or__(self, rhs):
		return Expr(self.op, self.lhs, rhs)
	def __repr__(self):
		return "PartialExpr('{}', {})".format(self.op, self.lhs)
def expr(x):
	if isinstance(x, str):
		return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol))
	else:
		return x
infix_ops = '==> <== <=>'.split()
def expr_handle_infix_ops(x):
	for op in infix_ops:
		x = x.replace(op, '|' + repr(op) + '|')
	return x
class defaultkeydict(collections.defaultdict):
	def __missing__(self, key):
		self[key] = result = self.default_factory(key)
		return result
class hashabledict(dict):
	def __hash__(self):
		return 1
class PriorityQueue:
	def __init__(self, order='min', f=lambda x: x):
		self.heap = []
		if order == 'min':
			self.f = f
		elif order == 'max': # now item with max f(x)
			self.f = lambda x: -f(x) # will be popped first
		else:
			raise ValueError("order must be either 'min' or max'.")
	def append(self, item):
		heapq.heappush(self.heap, (self.f(item), item))
	def extend(self, items):
		for item in items:
			self.heap.append(item)
	def pop(self):
		if self.heap:
			return heapq.heappop(self.heap)[1]
		else:
			raise Exception('Trying to pop from empty PriorityQueue.')
	def __len__(self):
		return len(self.heap)
	def __contains__(self, item):
		return (self.f(item), item) in self.heap
	def __getitem__(self, key):
		for _, item in self.heap:
			if item == key:
				return item
	def __delitem__(self, key):
		self.heap.remove((self.f(key), key))
		heapq.heapify(self.heap)
# ______________________________________________________________________________
# Useful Shorthands
class Bool(int):
	__str__ = __repr__ = lambda self: 'T' if self else 'F'
